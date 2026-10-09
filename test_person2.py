"""QueueFlux — Person 2 Test Suite
=================================
Automated unit and integration tests verifying all 4 modules owned by Person 2:
1. workload_model.py
2. state_estimator.py
3. wait_predictor.py
4. forecast_engine.py

Verifies shared interfaces, mathematical formulas, and mock data capabilities.
"""

import unittest
from typing import Any, Dict, List

from forecast_engine import (
    FiveMinuteProbabilisticForecaster,
    Person2PredictionPipeline,
    predict_queue_state,
)
from state_estimator import (
    EventCountBaselineEstimator,
    ParticleFilterQueueEstimator,
    SharedQueueState,
    StateEstimateResult,
    generate_mock_events,
)
from wait_predictor import WaitingTimePrediction, WorkloadWaitPredictor
from workload_model import (
    DEFAULT_BILLING_PARAMS,
    DEFAULT_KITCHEN_PARAMS,
    OrderType,
    ServiceDistributionParams,
    WorkloadModel,
)


class TestWorkloadModel(unittest.TestCase):
    """Verify workload calculations, lognormal parameters, and renewal residual times."""

    def setUp(self) -> None:
        self.wm = WorkloadModel()

    def test_order_type_category_workload_ordering(self) -> None:
        """Expected workloads must strictly follow: Beverage < Snack < Meal < Combo."""
        w_bev = self.wm.get_order_expected_workload(OrderType.BEVERAGE)
        w_snack = self.wm.get_order_expected_workload(OrderType.SNACK)
        w_meal = self.wm.get_order_expected_workload(OrderType.MEAL)
        w_combo = self.wm.get_order_expected_workload(OrderType.COMBO)

        self.assertLess(w_bev, w_snack)
        self.assertLess(w_snack, w_meal)
        self.assertLess(w_meal, w_combo)
        self.assertAlmostEqual(w_bev, 1.2, places=1)
        self.assertAlmostEqual(w_meal, 5.0, places=1)

    def test_renewal_residual_service_workload(self) -> None:
        """Residual service times must follow E[R] = (mean/2) * (1 + CV^2)."""
        params = ServiceDistributionParams(mean_minutes=2.0, sigma=0.3)
        expected_cv = params.std_dev / params.mean_minutes
        expected_r = (2.0 / 2.0) * (1.0 + expected_cv ** 2)
        self.assertAlmostEqual(params.mean_residual_service_time, expected_r, places=3)

        # Check total residual for 3 active kitchen servers
        res_3 = self.wm.get_residual_service_workload(active_servers=3, stage="kitchen")
        self.assertGreater(res_3, 0.0)

    def test_queue_workload_estimation(self) -> None:
        """Workload from queue counts must equal queue_count * expected_duration."""
        w_billing_10 = self.wm.estimate_stage_workload(queue_count=10, stage="billing")
        self.assertAlmostEqual(w_billing_10, 10 * 0.75, places=2)

        w_zero = self.wm.estimate_stage_workload(queue_count=0, stage="kitchen")
        self.assertEqual(w_zero, 0.0)

    def test_calculate_workload_from_order_list(self) -> None:
        """Exact order list workload summation."""
        orders = [
            {"order_type": "BEVERAGE"},
            {"order_type": "MEAL"},
            {"order_type": "COMBO"},
        ]
        total = self.wm.calculate_workload_from_orders(orders, stage="kitchen")
        expected = 1.2 + 5.0 + 7.5
        self.assertAlmostEqual(total, expected, places=2)


class TestStateEstimator(unittest.TestCase):
    """Verify Bayesian state estimation, shared state format, and credible intervals."""

    def setUp(self) -> None:
        self.estimator = ParticleFilterQueueEstimator(
            num_particles=80,
            active_billing_counters=2,
            active_kitchen_servers=3,
            seed=42,
        )

    def test_shared_state_format_compliance(self) -> None:
        """Estimator output must strictly adhere to the agreed shared state format."""
        events = generate_mock_events(start_t_minutes=0.0, count=4)
        self.estimator.predict(1.0)
        result = self.estimator.update_from_events(events)

        state_dict = result.state.to_dict()
        required_keys = {
            "billing_queue",
            "kitchen_queue",
            "billing_workload_min",
            "kitchen_workload_min",
            "active_billing_counters",
            "active_kitchen_servers",
        }
        self.assertTrue(required_keys.issubset(state_dict.keys()))
        self.assertIsInstance(state_dict["billing_queue"], int)
        self.assertIsInstance(state_dict["kitchen_queue"], int)
        self.assertIsInstance(state_dict["billing_workload_min"], float)
        self.assertIsInstance(state_dict["kitchen_workload_min"], float)

    def test_credible_intervals_validity(self) -> None:
        """Upper credible interval bound must be >= lower bound."""
        self.estimator.predict(1.0)
        result = self.estimator.update_from_events([], noisy_camera_billing_count=5)

        self.assertGreaterEqual(result.q_billing_ci_90[1], result.q_billing_ci_90[0])
        self.assertGreaterEqual(result.q_kitchen_ci_90[1], result.q_kitchen_ci_90[0])
        self.assertGreaterEqual(result.w_billing_ci_90[1], result.w_billing_ci_90[0])
        self.assertGreaterEqual(result.uncertainty_score, 0.0)
        self.assertLessEqual(result.uncertainty_score, 1.0)

    def test_baseline_estimator_comparison(self) -> None:
        """EventCountBaselineEstimator processes events and outputs shared format."""
        baseline = EventCountBaselineEstimator(active_billing=2, active_kitchen=3)
        events = [
            {"event_type": "ARRIVED"},
            {"event_type": "ARRIVED"},
            {"event_type": "BILLING_COMPLETED"},
        ]
        state = baseline.process_events(events)
        self.assertEqual(state.billing_queue, 1)
        self.assertEqual(state.kitchen_queue, 1)
        self.assertGreater(state.billing_workload_min, 0.0)
        self.assertGreater(state.kitchen_workload_min, 0.0)


class TestWaitPredictor(unittest.TestCase):
    """Verify registration-to-collection waiting time predictions."""

    def setUp(self) -> None:
        self.predictor = WorkloadWaitPredictor()
        self.state = SharedQueueState(
            billing_queue=4,
            kitchen_queue=6,
            billing_workload_min=3.0,
            kitchen_workload_min=18.0,
            active_billing_counters=2,
            active_kitchen_servers=3,
        )

    def test_order_category_waiting_time_monotonicity(self) -> None:
        """Combo meal must have strictly higher wait time than beverage in same queue."""
        pred_bev = self.predictor.predict_waiting_time(OrderType.BEVERAGE, self.state)
        pred_meal = self.predictor.predict_waiting_time(OrderType.MEAL, self.state)
        pred_combo = self.predictor.predict_waiting_time(OrderType.COMBO, self.state)

        self.assertLess(pred_bev.expected_total_wait_min, pred_meal.expected_total_wait_min)
        self.assertLess(pred_meal.expected_total_wait_min, pred_combo.expected_total_wait_min)

    def test_server_capacity_reduces_waiting_time(self) -> None:
        """Adding kitchen servers must reduce waiting time for same workload."""
        pred_3_servers = self.predictor.predict_waiting_time(OrderType.MEAL, self.state, active_kitchen_servers=3)
        pred_5_servers = self.predictor.predict_waiting_time(OrderType.MEAL, self.state, active_kitchen_servers=5)

        self.assertLess(pred_5_servers.expected_total_wait_min, pred_3_servers.expected_total_wait_min)

    def test_prediction_intervals_and_exceedance(self) -> None:
        """Prediction bounds and exceedance probabilities must be physically consistent."""
        pred = self.predictor.predict_waiting_time(OrderType.MEAL, self.state)
        self.assertGreaterEqual(pred.upper_bound_90_min, pred.expected_total_wait_min)
        self.assertLessEqual(pred.lower_bound_10_min, pred.expected_total_wait_min)
        self.assertGreaterEqual(pred.exceedance_probability_15min, 0.0)
        self.assertLessEqual(pred.exceedance_probability_15min, 1.0)


class TestForecastEngineAndIntegration(unittest.TestCase):
    """Verify 5-minute probabilistic forecasts and Person 2 unified interface."""

    def test_five_minute_forecast_bounds(self) -> None:
        """Forecaster outputs valid medians, intervals, and bottleneck probability."""
        pipeline = Person2PredictionPipeline(active_billing_counters=2, active_kitchen_servers=3, seed=123)
        mock_events = generate_mock_events(0.0, 1.0, count=5)

        output = pipeline.predict_queue_state(observed_events=mock_events, delta_time_minutes=1.0)

        fc = output["five_minute_forecast"]
        self.assertEqual(fc["forecast_horizon_min"], 5.0)
        self.assertGreaterEqual(fc["median_billing_queue"], 0.0)
        self.assertGreaterEqual(fc["median_kitchen_queue"], 0.0)
        self.assertGreaterEqual(fc["bottleneck_probability"], 0.0)
        self.assertLessEqual(fc["bottleneck_probability"], 1.0)
        self.assertGreaterEqual(fc["pi_90_billing_queue"][1], fc["pi_90_billing_queue"][0])

    def test_person2_callable_interface(self) -> None:
        """Test module-level callable `predict_queue_state` required by Section 3."""
        mock_events = [
            {"timestamp": 60, "event_type": "ARRIVED", "order_type": "MEAL"},
            {"timestamp": 90, "event_type": "BILLING_COMPLETED", "order_type": "MEAL"},
        ]
        result = predict_queue_state(observed_events=mock_events, delta_time_minutes=1.0, noisy_camera_count=3)

        self.assertIn("shared_state", result)
        self.assertIn("waiting_time_predictions", result)
        self.assertIn("five_minute_forecast", result)
        self.assertIn("MEAL", result["waiting_time_predictions"])
        self.assertIn("BEVERAGE", result["waiting_time_predictions"])


if __name__ == "__main__":
    unittest.main()
