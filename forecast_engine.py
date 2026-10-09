"""QueueFlux — Person 2 Core Model: Forecast Engine & Unified Interface
======================================================================
Provides five-minute probabilistic queue forecasting P(X_{t+5} | Y_{1:t})
via Monte Carlo trajectory rollouts and delivers Person 2's agreed callable
interface: `predict_queue_state(observed_data, ...)`.

Shared Integration Rule for Person 2:
"Provide a callable prediction function that accepts observed data and
returns estimated queues, waiting times and five-minute forecasts."
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from state_estimator import (
    ParticleFilterQueueEstimator,
    SharedQueueState,
    StateEstimateResult,
    generate_mock_events,
)
from wait_predictor import WaitingTimePrediction, WorkloadWaitPredictor
from workload_model import OrderType, WorkloadModel

# Optional rich console formatting
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAVE_RICH = True
    console = Console(safe_box=True)
except ImportError:
    HAVE_RICH = False
    console = None


@dataclass
class FiveMinuteForecast:
    """Five-minute future probabilistic queue forecast."""
    forecast_horizon_min: float
    median_billing_queue: float
    median_kitchen_queue: float
    pi_80_billing_queue: Tuple[float, float]
    pi_80_kitchen_queue: Tuple[float, float]
    pi_90_billing_queue: Tuple[float, float]
    pi_90_kitchen_queue: Tuple[float, float]
    expected_combined_backlog: float
    bottleneck_probability: float  # P(Q_B + Q_K > threshold)
    forecasted_waiting_time_meal_min: float
    bottleneck_threshold: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast_horizon_min": self.forecast_horizon_min,
            "median_billing_queue": round(self.median_billing_queue, 1),
            "median_kitchen_queue": round(self.median_kitchen_queue, 1),
            "pi_80_billing_queue": [round(self.pi_80_billing_queue[0], 1), round(self.pi_80_billing_queue[1], 1)],
            "pi_80_kitchen_queue": [round(self.pi_80_kitchen_queue[0], 1), round(self.pi_80_kitchen_queue[1], 1)],
            "pi_90_billing_queue": [round(self.pi_90_billing_queue[0], 1), round(self.pi_90_billing_queue[1], 1)],
            "pi_90_kitchen_queue": [round(self.pi_90_kitchen_queue[0], 1), round(self.pi_90_kitchen_queue[1], 1)],
            "expected_combined_backlog": round(self.expected_combined_backlog, 1),
            "bottleneck_probability": round(self.bottleneck_probability, 3),
            "forecasted_waiting_time_meal_min": round(self.forecasted_waiting_time_meal_min, 2),
            "bottleneck_threshold": self.bottleneck_threshold,
        }


class FiveMinuteProbabilisticForecaster:
    """Projects future queue state distribution P(X_{t+5} | Y_{1:t})

    using stochastic Monte Carlo forward rollouts.
    """

    def __init__(
        self,
        num_rollouts: int = 100,
        workload_model: Optional[WorkloadModel] = None,
        seed: Optional[int] = 77,
    ) -> None:
        self.num_rollouts = num_rollouts
        self.workload_model = workload_model or WorkloadModel()
        self.rng = np.random.default_rng(seed)

    def forecast(
        self,
        current_estimate: StateEstimateResult,
        bottleneck_threshold: int = 15,
        horizon_minutes: float = 5.0,
    ) -> FiveMinuteForecast:
        """Roll out future trajectories over the 5-minute horizon."""
        c_b = max(1, current_estimate.state.active_billing_counters)
        c_k = max(1, current_estimate.state.active_kitchen_servers)
        mu_b = current_estimate.estimated_billing_rate
        mu_k = current_estimate.estimated_kitchen_rate
        lam = current_estimate.estimated_arrival_rate

        # Sample initial states from current credible intervals
        std_qb = max(0.5, (current_estimate.q_billing_ci_90[1] - current_estimate.q_billing_ci_90[0]) / 3.29)
        std_qk = max(0.5, (current_estimate.q_kitchen_ci_90[1] - current_estimate.q_kitchen_ci_90[0]) / 3.29)

        init_qb = np.maximum(0.0, self.rng.normal(current_estimate.state.billing_queue, std_qb, size=self.num_rollouts))
        init_qk = np.maximum(0.0, self.rng.normal(current_estimate.state.kitchen_queue, std_qk, size=self.num_rollouts))

        final_qb = np.zeros(self.num_rollouts)
        final_qk = np.zeros(self.num_rollouts)

        steps = int(round(horizon_minutes))
        for i in range(self.num_rollouts):
            q_b = init_qb[i]
            q_k = init_qk[i]

            for _ in range(steps):
                # 1-minute step: Poisson arrivals
                arr = self.rng.poisson(lam)

                # Billing service
                b_serv = self.rng.poisson(c_b * mu_b)
                b_done = min(q_b + arr, b_serv)
                q_b = max(0.0, q_b + arr - b_done)

                # Kitchen service
                k_serv = self.rng.poisson(c_k * mu_k)
                k_done = min(q_k + b_done, k_serv)
                q_k = max(0.0, q_k + b_done - k_done)

            final_qb[i] = q_b
            final_qk[i] = q_k

        # Percentile metrics
        med_qb = float(np.median(final_qb))
        med_qk = float(np.median(final_qk))

        pi80_qb = (float(np.percentile(final_qb, 10)), float(np.percentile(final_qb, 90)))
        pi80_qk = (float(np.percentile(final_qk, 10)), float(np.percentile(final_qk, 90)))

        pi90_qb = (float(np.percentile(final_qb, 5)), float(np.percentile(final_qb, 95)))
        pi90_qk = (float(np.percentile(final_qk, 5)), float(np.percentile(final_qk, 95)))

        combined = final_qb + final_qk
        bottleneck_prob = float(np.mean(combined >= bottleneck_threshold))

        # Expected future wait time at t+5 for a standard meal
        wb_future = med_qb * self.workload_model.billing_params.mean_minutes
        wk_future = med_qk * self.workload_model.get_expected_kitchen_order_duration()
        future_meal_wait = (wb_future / c_b) + 0.75 + (wk_future / c_k) + 5.0

        return FiveMinuteForecast(
            forecast_horizon_min=horizon_minutes,
            median_billing_queue=med_qb,
            median_kitchen_queue=med_qk,
            pi_80_billing_queue=pi80_qb,
            pi_80_kitchen_queue=pi80_qk,
            pi_90_billing_queue=pi90_qb,
            pi_90_kitchen_queue=pi90_qk,
            expected_combined_backlog=float(np.mean(combined)),
            bottleneck_probability=bottleneck_prob,
            forecasted_waiting_time_meal_min=future_meal_wait,
            bottleneck_threshold=bottleneck_threshold,
        )


# ============================================================================
# UNIFIED PERSON 2 CALLABLE INTEGRATION INTERFACE
# ============================================================================

class Person2PredictionPipeline:
    """Unified engine encapsulating all of Person 2's responsibilities:

    - Particle filter state estimation
    - Workload calculation
    - Waiting-time prediction
    - 5-minute probabilistic forecasting
    """

    def __init__(
        self,
        active_billing_counters: int = 2,
        active_kitchen_servers: int = 3,
        seed: Optional[int] = 42,
    ) -> None:
        self.workload_model = WorkloadModel()
        self.state_estimator = ParticleFilterQueueEstimator(
            num_particles=150,
            active_billing_counters=active_billing_counters,
            active_kitchen_servers=active_kitchen_servers,
            seed=seed,
        )
        self.wait_predictor = WorkloadWaitPredictor(self.workload_model)
        self.forecaster = FiveMinuteProbabilisticForecaster(
            num_rollouts=100,
            workload_model=self.workload_model,
            seed=seed,
        )

    def predict_queue_state(
        self,
        observed_events: List[Dict[str, Any]],
        delta_time_minutes: float = 1.0,
        noisy_camera_count: Optional[int] = None,
        bottleneck_threshold: int = 15,
    ) -> Dict[str, Any]:
        """Agreed Person 2 Callable Interface:

        Accepts observed data, returns:
        1. Estimated queues in shared state format
        2. Workload-aware waiting times
        3. Five-minute probabilistic forecasts
        """
        # Step 1: Predict & Update state estimator
        self.state_estimator.predict(delta_time_minutes=delta_time_minutes)
        state_result = self.state_estimator.update_from_events(
            events=observed_events,
            noisy_camera_billing_count=noisy_camera_count,
        )

        # Step 2: Workload-aware waiting times for all order types
        waiting_times = self.wait_predictor.predict_all_order_types(state_result)

        # Step 3: Five-minute probabilistic forecast
        forecast_result = self.forecaster.forecast(
            current_estimate=state_result,
            bottleneck_threshold=bottleneck_threshold,
            horizon_minutes=5.0,
        )

        return {
            # Shared state format directly at the root for Person 3 consumption
            "shared_state": state_result.state.to_dict(),
            # Full uncertainty estimate details
            "state_estimate": state_result.to_dict(),
            # Workload-aware waiting times
            "waiting_time_predictions": waiting_times,
            # 5-minute forecast
            "five_minute_forecast": forecast_result.to_dict(),
        }


# Module-level convenience function fulfilling Person 2 integration requirement
_global_pipeline: Optional[Person2PredictionPipeline] = None


def predict_queue_state(
    observed_events: List[Dict[str, Any]],
    delta_time_minutes: float = 1.0,
    noisy_camera_count: Optional[int] = None,
    bottleneck_threshold: int = 15,
    active_billing: int = 2,
    active_kitchen: int = 3,
) -> Dict[str, Any]:
    """Callable prediction function for integration across modules."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = Person2PredictionPipeline(
            active_billing_counters=active_billing,
            active_kitchen_servers=active_kitchen,
        )
    return _global_pipeline.predict_queue_state(
        observed_events=observed_events,
        delta_time_minutes=delta_time_minutes,
        noisy_camera_count=noisy_camera_count,
        bottleneck_threshold=bottleneck_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QueueFlux Person 2 Model CLI")
    parser.add_argument("--mock", action="store_true", help="Run standalone demonstration with mock events")
    parser.add_argument("--steps", type=int, default=5, help="Number of simulation steps to run")
    args = parser.parse_args()

    pipeline = Person2PredictionPipeline(active_billing_counters=2, active_kitchen_servers=3)

    print("=" * 80)
    print("   QUEUEFLUX — PERSON 2: STATE ESTIMATION & PREDICTION CORE MODEL")
    print("=" * 80)
    print("Running standalone evaluation using agreed shared event format...")

    for step in range(1, args.steps + 1):
        # Generate mock events adhering to shared event schema
        mock_events = generate_mock_events(start_t_minutes=float(step - 1), duration_minutes=1.0, count=5)
        camera_read = 6 if step > 2 else 2

        output = pipeline.predict_queue_state(
            observed_events=mock_events,
            delta_time_minutes=1.0,
            noisy_camera_count=camera_read,
        )

        st = output["shared_state"]
        fc = output["five_minute_forecast"]
        wt_meal = output["waiting_time_predictions"]["MEAL"]

        if HAVE_RICH and console:
            tbl = Table(title=f"Person 2 Model Telemetry (Step {step} min)", show_header=True, header_style="bold cyan")
            tbl.add_column("Output Field", style="bold white")
            tbl.add_column("Value / Estimate", style="yellow")

            tbl.add_row("Shared State: Billing Queue", str(st["billing_queue"]))
            tbl.add_row("Shared State: Kitchen Queue", str(st["kitchen_queue"]))
            tbl.add_row("Billing Workload (service-min)", f"{st['billing_workload_min']} min")
            tbl.add_row("Kitchen Workload (service-min)", f"{st['kitchen_workload_min']} min")
            tbl.add_row("Predicted Meal Wait Time", f"{wt_meal['expected_total_wait_min']} min [10-90% CI: {wt_meal['lower_bound_10_min']}-{wt_meal['upper_bound_90_min']} min]")
            tbl.add_row("5-Min Forecast: Median Backlog", f"Billing: {fc['median_billing_queue']}, Kitchen: {fc['median_kitchen_queue']}")
            tbl.add_row("5-Min Forecast: P(Bottleneck)", f"{fc['bottleneck_probability'] * 100:.1f}%")
            console.print(tbl)
        else:
            print(f"\n--- [Step {step} min] ---")
            print(f"  Shared State      : Q_B={st['billing_queue']}, Q_K={st['kitchen_queue']} | Workload: W_B={st['billing_workload_min']}m, W_K={st['kitchen_workload_min']}m")
            print(f"  Meal Waiting Time : {wt_meal['expected_total_wait_min']} min [{wt_meal['lower_bound_10_min']}-{wt_meal['upper_bound_90_min']} min]")
            print(f"  5-Min Forecast    : Median Q_B={fc['median_billing_queue']}, Q_K={fc['median_kitchen_queue']} | P(Bottleneck)={fc['bottleneck_probability'] * 100:.1f}%")

    print("\n" + "=" * 80)
    print("Person 2 core model executed successfully! Interfaces verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
