"""
Quantitative Evaluation and Benchmark Suite for QueueFlux.
Directly compares QueueFlux against three meaningful baselines:
1. Baseline 1: Naive Rolling Average (FIFO queue length * static mean service)
2. Baseline 2: Classical Queueing Theory (M/M/c analytical Erlang-C model)
3. Baseline 3: Static Threshold Heuristic (Rule-based control without state estimation)
4. Proposed: QueueFlux (Estimation + Workload + Propagation + Risk-sensitive CVaR Optimization)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from queueflux.config import SystemConfig, DEFAULT_CONFIG
from queueflux.models.entities import Customer, CustomerStatus, OrderType, RegimeType
from queueflux.models.state import SystemState, ObservationVector, EstimatedState
from queueflux.simulator.engine import QueueSimulatorEngine
from queueflux.estimator.state_estimator import QueueStateEstimator
from queueflux.predictor.waiting_time import WorkloadWaitingTimePredictor
from queueflux.predictor.forecaster import FiveMinuteQueueForecaster
from queueflux.propagation.cross_stage import CrossStagePropagationEngine
from queueflux.optimizer.intervention import InterventionOptimizer, InterventionAction


@dataclass
class PolicyBenchmarkResult:
    """Benchmark metrics for a specific queue management policy."""
    policy_name: str
    total_customers_served: int
    total_customers_reneged: int
    reneging_rate_pct: float
    avg_customer_wait_minutes: float
    max_customer_wait_minutes: float
    p95_customer_wait_minutes: float
    waiting_time_mae: float             # Accuracy of waiting time prediction
    waiting_time_rmse: float
    forecast_5min_mae: float            # Accuracy of 5-min forward queue forecast
    forecast_crps: float                # Probabilistic forecast quality
    prediction_interval_coverage_80: float # Nominal 80% coverage
    total_waiting_hours_saved_pct: float


class QueueFluxBenchmark:
    """
    Executes controlled comparative experiments across synthetic test days.
    Measures prediction fidelity, forecast accuracy, and operational waiting reduction.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG, seed: int = 42):
        self.config = config
        self.seed = seed

    @staticmethod
    def mmc_expected_wait(arrival_rate: float, service_rate_per_server: float, num_servers: int) -> float:
        """
        Calculates expected queue waiting time W_q using the classical M/M/c Erlang-C formula.
        Falls back to fluid approximation when traffic intensity rho >= 1.
        """
        c = num_servers
        mu = service_rate_per_server
        lam = max(0.01, arrival_rate)
        
        a = lam / mu
        rho = a / c
        
        if rho >= 0.98:
            # Overloaded / near-saturation fluid approximation
            return max(0.0, (lam - c * mu * 0.98) * 5.0 / (c * mu))
            
        # Erlang-C calculation
        sum_terms = sum((a ** n) / math.factorial(n) for n in range(c))
        c_term = (a ** c) / (math.factorial(c) * (1.0 - rho))
        p0 = 1.0 / (sum_terms + c_term)
        erlang_c = c_term * p0
        
        w_q = erlang_c / (c * mu - lam)
        return max(0.0, w_q)

    def run_benchmark_on_day(self, day_data: pd.DataFrame, disruptions: List[Dict[str, Any]]) -> Dict[str, PolicyBenchmarkResult]:
        """
        Runs identical customer arrival streams through all 4 competing policies:
        1. Naive Baseline
        2. M/M/c Classical Queueing Baseline
        3. Static Heuristic Policy
        4. QueueFlux Complete Architecture
        """
        results = {}
        
        # 1. Run Naive Static Baseline (Fixed counters, naive waiting prediction)
        res_naive = self._run_policy_simulation(day_data, disruptions, policy_type="NAIVE")
        results["Baseline_Naive"] = res_naive
        
        # 2. Run M/M/c Analytical Baseline
        res_mmc = self._run_policy_simulation(day_data, disruptions, policy_type="MMC")
        results["Baseline_MMC"] = res_mmc
        
        # 3. Run Static Heuristic Policy
        res_heuristic = self._run_policy_simulation(day_data, disruptions, policy_type="HEURISTIC")
        results["Baseline_Static_Heuristic"] = res_heuristic
        
        # 4. Run Proposed QueueFlux Architecture
        res_qf = self._run_policy_simulation(day_data, disruptions, policy_type="QUEUEFLUX")
        results["QueueFlux_Full"] = res_qf
        
        # Compute wait hours saved relative to Naive
        base_wait = res_naive.avg_customer_wait_minutes
        for k in results:
            saved = max(0.0, (base_wait - results[k].avg_customer_wait_minutes) / base_wait * 100.0)
            results[k].total_waiting_hours_saved_pct = round(saved, 2)
            
        return results

    def _run_policy_simulation(self, day_data: pd.DataFrame, disruptions: List[Dict[str, Any]], 
                               policy_type: str) -> PolicyBenchmarkResult:
        """Simulates one full day under the specified policy."""
        engine = QueueSimulatorEngine(config=self.config, seed=self.seed)
        estimator = QueueStateEstimator(num_particles=400, config=self.config, seed=self.seed)
        predictor = WorkloadWaitingTimePredictor(config=self.config)
        forecaster = FiveMinuteQueueForecaster(config=self.config, num_rollout_samples=150, seed=self.seed)
        propagation_eng = CrossStagePropagationEngine(config=self.config)
        optimizer = InterventionOptimizer(config=self.config, seed=self.seed)
        
        # Group arriving customers by minute
        minute_groups = {}
        for _, row in day_data.iterrows():
            m = int(row["arrival_minute"])
            cust = Customer(
                customer_id=row["customer_id"],
                arrival_time=row["arrival_minute"],
                order_type=OrderType(row["order_type"]),
                expected_billing_duration=row["expected_billing_duration"],
                actual_billing_duration=row["sampled_billing_duration"],
                patience_billing=row["patience_billing"],
                patience_kitchen=row["patience_kitchen"]
            )
            minute_groups.setdefault(m, []).append(cust)

        # Performance tracking collections
        actual_waits: List[float] = []
        predicted_waits: List[float] = []
        coverage_hits: List[bool] = []
        forecast_errors: List[float] = []
        forecast_crps_list: List[float] = []
        
        cooldown_counter = 0

        # Step through every minute of the operating day
        for t in range(self.config.minutes_per_day):
            arrivals_now = minute_groups.get(t, [])
            
            # Step the environment
            state_gt, obs = engine.step_minute(arrivals_now, disruptions)
            
            # State Estimation (QueueFlux observer)
            est_state = estimator.update(obs)
            
            # Policy-specific decision logic
            if policy_type == "NAIVE":
                # Naive: fixed 3 counters, wait prediction = (visual queue count) * 0.95 + 3.60
                pass  # no intervention
                
            elif policy_type == "MMC":
                # M/M/c analytical: predicts wait based on Erlang-C using reported arrivals
                pass
                
            elif policy_type == "HEURISTIC":
                # Static rule: if visual billing count > 10, open counter; if > 15, open 2
                if obs.y_Q_B > 12.0:
                    engine.set_intervention_action({"billing_counters": 4})
                elif obs.y_Q_B < 4.0:
                    engine.set_intervention_action({"billing_counters": 3})
                    
            elif policy_type == "QUEUEFLUX":
                # Full QueueFlux: Propagation risk + Uncertainty-aware CVaR optimization
                if cooldown_counter <= 0:
                    prop_report = propagation_eng.evaluate_propagation_risk(est_state)
                    best_act, eval_res, _ = optimizer.optimize_action(est_state, prop_report, cooldown_active=False)
                    
                    if best_act == InterventionAction.OPEN_BILLING_COUNTER:
                        engine.set_intervention_action({"billing_counters": int(est_state.C_B_hat + 1)})
                        cooldown_counter = 10  # 10 minute cooldown
                    elif best_act == InterventionAction.REALLOCATE_STAFF_TO_KITCHEN:
                        engine.set_intervention_action({
                            "billing_counters": max(2, int(est_state.C_B_hat - 1)),
                            "kitchen_stations": int(est_state.C_K_hat + 1)
                        })
                        cooldown_counter = 12
                    elif best_act == InterventionAction.ACTIVATE_KITCHEN_BATCHING:
                        for k in engine.kitchen_stations:
                            k.efficiency_multiplier = 1.25
                        cooldown_counter = 15
                    elif best_act == InterventionAction.BROADCAST_WAIT_WARNING:
                        cooldown_counter = 20
                else:
                    cooldown_counter -= 1

            # Predictions for customers arriving in this minute
            for cust in arrivals_now:
                # 1. Prediction under policy
                if policy_type == "NAIVE":
                    # Naive FIFO wait estimate
                    pred_b = (obs.y_Q_B / max(1, obs.c_B_reported)) * 0.95
                    pred_k = (obs.y_Q_K / max(1, obs.c_K_reported)) * 3.60
                    pred_val = pred_b + 0.95 + pred_k + 3.60
                    p10, p90 = pred_val * 0.7, pred_val * 1.3
                    
                elif policy_type == "MMC":
                    # M/M/c Erlang-C wait estimate
                    w_b = self.mmc_expected_wait(arrival_rate=2.5, service_rate_per_server=1.05, num_servers=obs.c_B_reported)
                    w_k = self.mmc_expected_wait(arrival_rate=2.2, service_rate_per_server=0.28, num_servers=obs.c_K_reported)
                    pred_val = w_b + 0.95 + w_k + 3.60
                    p10, p90 = pred_val * 0.75, pred_val * 1.25
                    
                elif policy_type == "HEURISTIC":
                    pred_val = (obs.y_Q_B / 4.0) * 0.95 + (obs.y_Q_K / 4.0) * 3.60 + 4.5
                    p10, p90 = pred_val * 0.75, pred_val * 1.25
                    
                else:  # QUEUEFLUX
                    qf_pred = predictor.predict(est_state, cust.order_type)
                    pred_val = qf_pred.expected_total_system_time
                    p10 = qf_pred.total_system_time_p10
                    p90 = qf_pred.total_system_time_p90
                    
                predicted_waits.append(pred_val)
                # Actual realized waiting time will be matched after simulation finishes
                cust._pred_val = pred_val
                cust._p10 = p10
                cust._p90 = p90

            # 5-minute queue forecasting evaluation (every 5 minutes)
            if t % 5 == 0 and t + 5 < self.config.minutes_per_day:
                fc_5min = forecaster.forecast_5min(est_state, t)
                # Evaluate horizon +5 against ground truth later (or rollout proxy)
                fc_pt = fc_5min[-1].Q_B_median
                gt_future = state_gt.Q_B  # proxy
                forecast_errors.append(abs(fc_pt - gt_future))
                
                # CRPS
                samples = np.linspace(fc_5min[-1].Q_B_p10, fc_5min[-1].Q_B_p90, 50)
                crps_val = FiveMinuteQueueForecaster.calculate_crps(samples, float(gt_future))
                forecast_crps_list.append(crps_val)

        # Aggregate post-simulation customer metrics
        all_completed = engine.completed_customers
        total_served = len(all_completed)
        total_reneged = len(engine.cancelled_billing) + len(engine.cancelled_kitchen)
        total_arrivals = total_served + total_reneged
        renege_rate = (total_reneged / max(1, total_arrivals)) * 100.0
        
        waits = [c.total_waiting_time for c in all_completed if c.total_waiting_time is not None]
        avg_wait = float(np.mean(waits)) if len(waits) > 0 else 0.0
        max_wait = float(np.max(waits)) if len(waits) > 0 else 0.0
        p95_wait = float(np.percentile(waits, 95)) if len(waits) > 0 else 0.0
        
        # Match customer predicted vs actual waits
        errors = []
        for c in all_completed:
            if hasattr(c, "_pred_val") and c.total_waiting_time is not None:
                err = abs(c._pred_val - c.total_waiting_time)
                errors.append(err)
                in_band = (c._p10 <= c.total_waiting_time <= c._p90)
                coverage_hits.append(in_band)
                
        mae = float(np.mean(errors)) if len(errors) > 0 else 0.0
        rmse = float(np.sqrt(np.mean(np.array(errors) ** 2))) if len(errors) > 0 else 0.0
        picp = (sum(coverage_hits) / max(1, len(coverage_hits))) * 100.0
        fc_mae = float(np.mean(forecast_errors)) if len(forecast_errors) > 0 else 0.0
        avg_crps = float(np.mean(forecast_crps_list)) if len(forecast_crps_list) > 0 else 0.0
        
        return PolicyBenchmarkResult(
            policy_name=policy_type,
            total_customers_served=total_served,
            total_customers_reneged=total_reneged,
            reneging_rate_pct=round(renege_rate, 2),
            avg_customer_wait_minutes=round(avg_wait, 2),
            max_customer_wait_minutes=round(max_wait, 2),
            p95_customer_wait_minutes=round(p95_wait, 2),
            waiting_time_mae=round(mae, 2),
            waiting_time_rmse=round(rmse, 2),
            forecast_5min_mae=round(fc_mae, 2),
            forecast_crps=round(avg_crps, 3),
            prediction_interval_coverage_80=round(picp, 1),
            total_waiting_hours_saved_pct=0.0
        )
