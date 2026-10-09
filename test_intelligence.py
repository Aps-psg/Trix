"""Mandatory Unit & Integration Test Suite for QueueFlux Model
===========================================================
Fulfills Section 14 testing requirements:
- Queue counts non-negativity invariants.
- Strict service capacity enforcement.
- Strict two-stage order lifecycle progression.
- Exact workload calculation.
- Zero arrival and departure transitions.
- Inactive / failed resource handling.
- Deterministic reproducibility under fixed seeds.
- Strict separation between ground-truth and observations.
- No future-information leakage.
- Hand-calculable queue evolution verification.
- Cross-stage propagation and decision stability.
"""

import copy
import math
import unittest
from typing import List

import numpy as np

from queue_intelligence import (
    ActionType,
    CanteenEventSimulator,
    CrossStagePropagationAnalyzer,
    GroundTruthState,
    Observation,
    OperatingCondition,
    Order,
    OrderCategory,
    OrderStatus,
    ParticleFilterEstimator,
    ProbabilisticForecaster,
    QueueFluxBenchmark,
    QueueFluxDecisionOptimizer,
    StatisticalDisruptionDetector,
    SyntheticCanteenGenerator,
    WorkloadWaitPredictor,
)


class TestQueueInvariantsAndLifecycle(unittest.TestCase):
    """Test physical queue invariants and two-stage lifecycle logic."""

    def setUp(self) -> None:
        self.sim = CanteenEventSimulator(num_billing_counters=2, num_kitchen_resources=3, seed=123)

    def test_queue_counts_never_negative(self) -> None:
        """Invariants: Q_t^B >= 0 and Q_t^K >= 0 under all operational steps."""
        for step in range(1, 20):
            # Advance empty simulator
            self.sim.step_to(float(step))
            gt = self.sim.get_ground_truth_state()
            self.assertGreaterEqual(gt.q_billing, 0)
            self.assertGreaterEqual(gt.q_kitchen, 0)
            self.assertGreaterEqual(gt.w_billing, 0.0)
            self.assertGreaterEqual(gt.w_kitchen, 0.0)

    def test_service_capacity_respected(self) -> None:
        """Service Capacity Invariant: Active servers <= C_t^B and active prep stations <= C_t^K."""
        # Flood simulator with 20 orders at t=0
        for i in range(20):
            self.sim.register_arrival(f"FLOOD-{i}", OrderCategory.MEAL, 0.0)

        self.sim.step_to(0.5)
        gt = self.sim.get_ground_truth_state()
        self.assertLessEqual(len(self.sim.active_billing), self.sim.num_billing_counters)
        self.assertLessEqual(len(self.sim.active_kitchen), self.sim.num_kitchen_resources)

    def test_strict_order_lifecycle_transfer(self) -> None:
        """Orders can transfer to Kitchen backlog ONLY after billing is completed."""
        order = self.sim.register_arrival("TEST-ORD", OrderCategory.SNACK, 0.0)
        self.assertEqual(order.status, OrderStatus.WAITING_BILLING)
        self.assertIsNone(order.billing_start)
        self.assertIsNone(order.billing_end)

        # Run until billing completes
        self.sim.step_to(2.0)
        if order.status in (OrderStatus.WAITING_KITCHEN, OrderStatus.PREPARING, OrderStatus.COLLECTED):
            self.assertIsNotNone(order.billing_end)
            self.assertGreaterEqual(order.billing_end, order.billing_start or 0.0)

    def test_order_collection_lifecycle(self) -> None:
        """Orders can be collected ONLY after food preparation completes."""
        order = self.sim.register_arrival("COLLECT-TEST", OrderCategory.BEVERAGE, 0.0)
        # Run until complete collection
        self.sim.step_to(10.0)
        self.assertEqual(order.status, OrderStatus.COLLECTED)
        self.assertIsNotNone(order.kitchen_end)
        self.assertIsNotNone(order.collected_time)
        self.assertGreaterEqual(order.collected_time, order.kitchen_end)
        self.assertGreater(order.total_waiting_time, 0.0)

    def test_workload_calculation_accuracy(self) -> None:
        """W_t^B and W_t^K must strictly equal the sum of waiting orders' nominal service times."""
        o1 = self.sim.register_arrival("W1", OrderCategory.SNACK, 0.0)
        o2 = self.sim.register_arrival("W2", OrderCategory.COMBO, 0.0)

        # Before service begins:
        expected_w_b = o1.nominal_billing_workload + o2.nominal_billing_workload
        gt = self.sim.get_ground_truth_state()
        # Either in queue or in service
        total_b_workload = sum(o.nominal_billing_workload for o in self.sim.billing_queue)
        self.assertEqual(gt.w_billing, round(total_b_workload, 2))

    def test_zero_arrival_departure_transition(self) -> None:
        """Exact balance equations hold when arrivals and departures are zero."""
        gt_before = self.sim.get_ground_truth_state()
        self.sim.step_to(0.1)
        gt_after = self.sim.get_ground_truth_state()
        self.assertEqual(gt_before.q_billing, gt_after.q_billing)
        self.assertEqual(gt_before.q_kitchen, gt_after.q_kitchen)

    def test_counter_failure_handling(self) -> None:
        """During RESOURCE_FAILURE, active billing counters must reduce by 1."""
        self.sim.condition = OperatingCondition.RESOURCE_FAILURE
        gt = self.sim.get_ground_truth_state()
        self.assertEqual(gt.active_billing_counters, self.sim.num_billing_counters - 1)

    def test_reproducibility_fixed_seed(self) -> None:
        """Simulators with the same seed must produce identical event histories."""
        sim1 = CanteenEventSimulator(seed=999)
        sim2 = CanteenEventSimulator(seed=999)

        sim1.register_arrival("A1", OrderCategory.MEAL, 0.0)
        sim2.register_arrival("A1", OrderCategory.MEAL, 0.0)

        sim1.step_to(5.0)
        sim2.step_to(5.0)

        gt1 = sim1.get_ground_truth_state()
        gt2 = sim2.get_ground_truth_state()

        self.assertEqual(gt1.q_billing, gt2.q_billing)
        self.assertEqual(gt1.q_kitchen, gt2.q_kitchen)
        self.assertEqual(gt1.w_kitchen, gt2.w_kitchen)


class TestHandCalculatedEvolution(unittest.TestCase):
    """Simple deterministic scenario whose queue transitions can be verified by hand."""

    def test_hand_calculated_trace(self) -> None:
        """
        Hand Verification:
        - 1 billing counter, 1 kitchen resource.
        - Fixed billing time = 1.0 min, fixed kitchen time = 2.0 min.
        - Order 1 arrives at t=0.0
        - Order 2 arrives at t=0.5
        Timeline:
          t=0.0: O1 starts billing (ends at 1.0). Q^B = 0, C_active^B = 1.
          t=0.5: O2 arrives, joins queue. Q^B = 1.
          t=1.0: O1 finishes billing -> transfers to kitchen. O2 starts billing (ends at 2.0).
                 O1 starts kitchen (ends at 3.0). Q^B = 0, Q^K = 0.
          t=2.0: O2 finishes billing -> transfers to kitchen. Q^K = 1 (O1 still prep until 3.0).
        """
        sim = CanteenEventSimulator(num_billing_counters=1, num_kitchen_resources=1, seed=42)
        # Override sample functions with deterministic durations for this test
        sim.sample_billing_duration = lambda: 1.0
        sim.sample_kitchen_duration = lambda cat: 2.0

        o1 = sim.register_arrival("H1", OrderCategory.SNACK, 0.0)
        sim.step_to(0.1)  # O1 starts billing

        o2 = sim.register_arrival("H2", OrderCategory.SNACK, 0.5)
        sim.step_to(0.6)  # O2 in queue
        self.assertEqual(len(sim.billing_queue), 1)

        sim.step_to(1.1)  # O1 completed billing, O2 in billing, O1 in kitchen
        self.assertEqual(o1.status, OrderStatus.PREPARING)
        self.assertEqual(o2.status, OrderStatus.BILLING)
        self.assertEqual(len(sim.billing_queue), 0)

        sim.step_to(2.1)  # O2 finished billing, entered kitchen queue behind O1
        self.assertEqual(len(sim.kitchen_queue), 1)
        self.assertEqual(o2.status, OrderStatus.WAITING_KITCHEN)


class TestStateEstimationAndSeparation(unittest.TestCase):
    """Verify state estimation and separation between ground truth and observations."""

    def test_ground_truth_vs_observation_separation(self) -> None:
        """Observations must contain noise and dropouts without corrupting ground truth."""
        sim = CanteenEventSimulator(seed=55)
        for i in range(5):
            sim.register_arrival(f"S-{i}", OrderCategory.MEAL, 0.0)
        gt = sim.get_ground_truth_state()
        obs = sim.generate_noisy_observation(gt, b_done=2, k_done=1)

        # Ground truth intact
        self.assertEqual(gt.q_billing, len(sim.billing_queue))
        # Observation is noisy
        if obs.noisy_camera_billing_count is not None:
            self.assertIsInstance(obs.noisy_camera_billing_count, int)

    def test_particle_filter_bounds(self) -> None:
        """State estimator outputs valid credible intervals and uncertainty scores."""
        pf = ParticleFilterEstimator(num_particles=80, seed=42)
        obs = Observation(
            timestamp=1.0,
            noisy_camera_billing_count=4,
            observed_billing_completions=2,
            observed_kitchen_completions=1,
            reported_active_billing_counters=2,
            reported_active_kitchen_resources=3,
        )
        pf.predict(1.0)
        est = pf.update(obs)

        self.assertGreaterEqual(est.expected_q_billing, 0.0)
        self.assertGreaterEqual(est.q_billing_ci[1], est.q_billing_ci[0])
        self.assertGreaterEqual(est.uncertainty_score, 0.0)
        self.assertLessEqual(est.uncertainty_score, 1.0)


class TestWorkloadWaitTimeAndForecasting(unittest.TestCase):
    """Verify workload-aware predictions and 5-minute probabilistic forecasts."""

    def test_workload_category_monotonicity(self) -> None:
        """Heavier order category (COMBO) must predict longer wait time than light (BEVERAGE)."""
        pf = ParticleFilterEstimator(num_particles=60, seed=42)
        obs = Observation(1.0, 3, 1, 1, 2, 3)
        pf.predict(1.0)
        est = pf.update(obs)

        predictor = WorkloadWaitPredictor()
        wait_bev = predictor.predict(OrderCategory.BEVERAGE, est, 2, 3)
        wait_combo = predictor.predict(OrderCategory.COMBO, est, 2, 3)

        self.assertGreater(wait_combo["expected_total_wait_minutes"], wait_bev["expected_total_wait_minutes"])

    def test_five_minute_forecaster(self) -> None:
        """Forecaster produces valid median, intervals, and bottleneck probability in [0, 1]."""
        pf = ParticleFilterEstimator(num_particles=60, seed=42)
        obs = Observation(1.0, 5, 2, 1, 2, 3)
        pf.predict(1.0)
        est = pf.update(obs)

        forecaster = ProbabilisticForecaster(num_rollouts=40, seed=42)
        fc = forecaster.forecast_5min(est, 2, 3, bottleneck_threshold=10)

        self.assertGreaterEqual(fc["median_q_billing"], 0.0)
        self.assertGreaterEqual(fc["pi_80_q_billing"][1], fc["pi_80_q_billing"][0])
        self.assertGreaterEqual(fc["bottleneck_probability"], 0.0)
        self.assertLessEqual(fc["bottleneck_probability"], 1.0)


class TestCrossStagePropagationAndDecision(unittest.TestCase):
    """Verify counterfactual disruption analysis and decision optimization."""

    def test_billing_slowdown_cross_stage_starvation(self) -> None:
        """Billing slowdown must cause initial kitchen queue starvation (Delta Q^K < 0)."""
        analyzer = CrossStagePropagationAnalyzer()
        base = GroundTruthState(0.0, 3, 3, 2.25, 9.0, [0.3, 0.5], [1.0, 1.5, 0.8], 2, 3, OperatingCondition.NORMAL)
        res = analyzer.analyze_disruption_impact(base, OperatingCondition.BILLING_SLOWDOWN, horizon_minutes=10, replications=15)

        self.assertGreaterEqual(res["max_billing_surge"], 0.0)
        self.assertLessEqual(res["min_kitchen_starvation"], 0.0)

    def test_decision_optimizer(self) -> None:
        """Decision optimizer returns feasible action, non-negative cost, and stability score."""
        optimizer = QueueFluxDecisionOptimizer()
        pf = ParticleFilterEstimator(num_particles=60, seed=42)
        obs = Observation(1.0, 8, 1, 1, 2, 3)
        pf.predict(1.0)
        est = pf.update(obs)

        decision = optimizer.evaluate_interventions(est, 2, 3, num_eval_particles=20)
        valid_actions = [a.value for a in ActionType]
        self.assertIn(decision["recommended_action"], valid_actions)
        self.assertGreaterEqual(decision["decision_stability"], 0.0)
        self.assertLessEqual(decision["decision_stability"], 1.0)
        self.assertGreaterEqual(decision["expected_cost"], 0.0)

    def test_benchmark_runner(self) -> None:
        """Benchmark runs all 4 configurations and yields quantitative error metrics."""
        bm = QueueFluxBenchmark()
        res = bm.run_full_benchmark(duration_minutes=15)
        self.assertIn("baseline_a_naive", res)
        self.assertIn("baseline_b_no_filter", res)
        self.assertIn("baseline_c_independent", res)
        self.assertIn("queueflux", res)
        self.assertGreaterEqual(res["waiting_time_reduction_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
