"""
QueueFlux Interactive Demonstration Script.
Runs a live simulation of a college canteen rush day with an injected operational disruption,
demonstrating the 6 core pillars of QueueFlux:
1. State Estimation under Noisy Sensor Observations
2. Workload-Aware Waiting Time Prediction
3. 5-Minute Probabilistic Forecasting (P10, P50, P90)
4. Cross-Stage Disruption Propagation (Stage A -> Stage B)
5. Uncertainty-Aware Intervention Selection (CVaR optimization)
6. Real-time Comparative Metrics
"""

import sys
import time
import numpy as np
import pandas as pd

from queueflux.config import DEFAULT_CONFIG
from queueflux.models.entities import Customer, CustomerStatus, OrderType, RegimeType
from queueflux.models.state import SystemState, ObservationVector, EstimatedState
from queueflux.simulator.generator import TonicSyntheticDataGenerator
from queueflux.simulator.engine import QueueSimulatorEngine
from queueflux.estimator.state_estimator import QueueStateEstimator
from queueflux.predictor.waiting_time import WorkloadWaitingTimePredictor
from queueflux.predictor.forecaster import FiveMinuteQueueForecaster
from queueflux.propagation.cross_stage import CrossStagePropagationEngine
from queueflux.optimizer.intervention import InterventionOptimizer, InterventionAction


def run_interactive_demo():
    print("=" * 80)
    print(" QUEUEFLUX: PROPAGATION-AWARE, UNCERTAINTY-AWARE QUEUE OPTIMIZATION MODEL")
    print(" College Canteen Digital Twin & Predictive Decision Engine")
    print("=" * 80)
    print("Initializing environment and components...")

    config = DEFAULT_CONFIG
    generator = TonicSyntheticDataGenerator(config=config, seed=42)
    engine = QueueSimulatorEngine(config=config, seed=42)
    estimator = QueueStateEstimator(num_particles=400, config=config, seed=42)
    predictor = WorkloadWaitingTimePredictor(config=config)
    forecaster = FiveMinuteQueueForecaster(config=config, num_rollout_samples=150, seed=42)
    propagation = CrossStagePropagationEngine(config=config)
    optimizer = InterventionOptimizer(config=config, seed=42)

    # Generate synthetic day data for Day 4 (contains Billing Slowdown Disruption)
    print("\n[Step 1] Generating synthetic customer arrival stream (Tonic AI Schema - Day 4)...")
    day_df, disruptions = generator.generate_day(day_index=3)
    print(f"  Generated {len(day_df)} arriving customers across 12 operational hours (720 min).")
    print(f"  Active Disruption: {disruptions[0]['description']} from min {disruptions[0]['start_min']} to {disruptions[0]['end_min']}.")

    # Index arrivals by minute
    arrivals_by_min = {}
    for _, row in day_df.iterrows():
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
        arrivals_by_min.setdefault(m, []).append(cust)

    print("\n[Step 2] Simulating live canteen operational stream...")
    print("-" * 80)
    print(f"{'Time':<7} | {'GT Billing':<10} | {'Sensor Obs':<11} | {'Estimated Q_B (CI)':<18} | {'Est Regime':<11} | {'Recommended Action':<20}")
    print("-" * 80)

    # Focus observation output around lunch rush and disruption window: minute 260 to 310
    sample_windows = [60, 120, 240, 275, 285, 295, 305, 330, 540]
    
    cooldown = 0
    interventions_logged = []

    for t in range(config.minutes_per_day):
        arrs = arrivals_by_min.get(t, [])
        state_gt, obs = engine.step_minute(arrs, disruptions)
        est_state = estimator.update(obs)

        # Check propagation and optimizer
        prop_report = propagation.evaluate_propagation_risk(est_state)
        
        action_name = "DO_NOTHING"
        if cooldown <= 0:
            best_act, eval_res, _ = optimizer.optimize_action(est_state, prop_report)
            if best_act != InterventionAction.DO_NOTHING:
                action_name = best_act.value
                interventions_logged.append((t, best_act.value, eval_res.expected_wait_reduction_pct, eval_res.rationale))
                
                # Apply action to engine
                if best_act == InterventionAction.OPEN_BILLING_COUNTER:
                    engine.set_intervention_action({"billing_counters": int(est_state.C_B_hat + 1)})
                    cooldown = 10
                elif best_act == InterventionAction.REALLOCATE_STAFF_TO_KITCHEN:
                    engine.set_intervention_action({
                        "billing_counters": max(2, int(est_state.C_B_hat - 1)),
                        "kitchen_stations": int(est_state.C_K_hat + 1)
                    })
                    cooldown = 12
                elif best_act == InterventionAction.ACTIVATE_KITCHEN_BATCHING:
                    for k in engine.kitchen_stations:
                        k.efficiency_multiplier = 1.25
                    cooldown = 15
                elif best_act == InterventionAction.BROADCAST_WAIT_WARNING:
                    cooldown = 25
        else:
            cooldown -= 1

        # Print sampled snapshot telemetry
        if t in sample_windows:
            clock_hr = 8 + (t // 60)
            clock_mn = t % 60
            clock_str = f"{clock_hr:02d}:{clock_mn:02d}"
            
            ci_str = f"{est_state.Q_B_hat:.1f} [{est_state.Q_B_p10:.1f}-{est_state.Q_B_p90:.1f}]"
            reg_str = est_state.most_likely_regime.value
            
            print(f"{clock_str:<7} | Q_B={state_gt.Q_B:<6} | y_B={obs.y_Q_B:<8.1f} | {ci_str:<18} | {reg_str:<11} | {action_name:<20}")

    print("-" * 80)
    print("\n[Step 3] Detailed Snapshot: 5-Minute Probabilistic Forecaster (t = 285 min / 12:45 PM Peak)")
    print("-" * 80)
    
    # Pick minute 285 snapshot
    t_inspect = 285
    fc_list = forecaster.forecast_5min(est_state, t_inspect)
    
    print(f"{'Horizon':<10} | {'Billing Q_B [P10 - Median - P90]':<32} | {'Kitchen Q_K [P10 - Median - P90]':<32}")
    print("-" * 80)
    for fc in fc_list:
        b_str = f"[{fc.Q_B_p10:4.1f}  -  {fc.Q_B_median:4.1f}  -  {fc.Q_B_p90:4.1f}]"
        k_str = f"[{fc.Q_K_p10:4.1f}  -  {fc.Q_K_median:4.1f}  -  {fc.Q_K_p90:4.1f}]"
        print(f"t + {fc.horizon_minutes} min   | {b_str:<32} | {k_str:<32}")

    print("-" * 80)
    print("\n[Step 4] Cross-Stage Disruption Propagation Diagnostic:")
    prop_diag = propagation.evaluate_propagation_risk(est_state)
    print(f"  Downstream Starvation Risk:        {prop_diag.downstream_starvation_risk:.1%}")
    print(f"  Downstream Surge Shockwave Risk:   {prop_diag.downstream_surge_shockwave_risk:.1%}")
    print(f"  Upstream Backpressure Risk:        {prop_diag.upstream_backpressure_risk:.1%}")
    print(f"  Propagation Peak Sensitivity Lag:  {prop_diag.cross_stage_sensitivity_peak_lag} minutes")
    print(f"  Recommended Stage Focus:           {prop_diag.recommended_stage_focus}")

    print("\n[Step 5] Triggered Uncertainty-Aware Interventions during Day:")
    if interventions_logged:
        for t_act, act_name, red_pct, rat in interventions_logged:
            hr = 8 + (t_act // 60)
            mn = t_act % 60
            print(f"  [{hr:02d}:{mn:02d}] {act_name} -> Est. Delay Reduction: {red_pct:.1f}%")
            print(f"          Rationale: {rat}")
    else:
        print("  System operated steadily without triggering critical intervention thresholds.")

    print("\n[Step 6] Day Summary Performance Metrics:")
    total_completed = len(engine.completed_customers)
    total_reneged_b = len(engine.cancelled_billing)
    total_reneged_k = len(engine.cancelled_kitchen)
    waits = [c.total_waiting_time for c in engine.completed_customers if c.total_waiting_time is not None]
    
    print(f"  Total Customers Arrived:     {len(day_df)}")
    print(f"  Successfully Served:        {total_completed}")
    print(f"  Reneged Billing:            {total_reneged_b}")
    print(f"  Cancelled Kitchen:          {total_reneged_k}")
    print(f"  Average Customer Wait Time: {np.mean(waits):.2f} minutes")
    print(f"  95th Percentile Wait Time:  {np.percentile(waits, 95):.2f} minutes")
    print(f"  Maximum Wait Time:          {np.max(waits):.2f} minutes")
    print("=" * 80)
    print(" Interactive Demo Complete. Run 'python main.py --days 30 --visualize' for full 30-day benchmark.")
    print("=" * 80)


if __name__ == "__main__":
    run_interactive_demo()
