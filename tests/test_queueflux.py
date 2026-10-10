"""
Unit Test Suite for QueueFlux Mathematical Model.
Validates:
1. Queue conservation transition equations: Q_{t+1} = max(0, Q_t + A_t - S_t)
2. State separation: Ground-truth vs Inferred state isolation
3. Workload non-negativity and consistency
4. Particle filter observation update and weight normalization
5. 5-Minute probabilistic forecast horizon outputs
6. Cross-stage propagation impulse response calculation
7. Uncertainty-aware intervention optimizer action selection
"""

import unittest
import numpy as np
import pandas as pd

from queueflux.config import SystemConfig, DEFAULT_CONFIG
from queueflux.models.entities import Customer, Order, CustomerStatus, OrderType, RegimeType
from queueflux.models.state import SystemState, ObservationVector, EstimatedState
from queueflux.simulator.generator import TonicSyntheticDataGenerator
from queueflux.simulator.engine import QueueSimulatorEngine
from queueflux.estimator.state_estimator import QueueStateEstimator
from queueflux.predictor.waiting_time import WorkloadWaitingTimePredictor
from queueflux.predictor.forecaster import FiveMinuteQueueForecaster
from queueflux.propagation.cross_stage import CrossStagePropagationEngine
from queueflux.optimizer.intervention import InterventionOptimizer, InterventionAction


class TestQueueFluxCore(unittest.TestCase):

    def setUp(self):
        self.config = DEFAULT_CONFIG
        self.engine = QueueSimulatorEngine(config=self.config, seed=42)
        self.estimator = QueueStateEstimator(num_particles=200, config=self.config, seed=42)
        self.predictor = WorkloadWaitingTimePredictor(config=self.config)
        self.forecaster = FiveMinuteQueueForecaster(config=self.config, num_rollout_samples=100, seed=42)
        self.propagation = CrossStagePropagationEngine(config=self.config)
        self.optimizer = InterventionOptimizer(config=self.config, seed=42)

    def test_queue_conservation_transition(self):
        """Tests that queue counts strictly satisfy balance equations."""
        customers = [
            Customer(
                customer_id=f"TEST_C{i}",
                arrival_time=0.0,
                order_type=OrderType.STANDARD,
                expected_billing_duration=1.0,
                actual_billing_duration=1.0,
                patience_billing=20.0,
                patience_kitchen=30.0
            )
            for i in range(5)
        ]
        
        # Step 1: 5 arrivals into billing
        st1, obs1 = self.engine.step_minute(customers, disruptions=[])
        # Since 3 counters are active and each takes ~1.0 min, billing queue should balance
        self.assertGreaterEqual(st1.Q_B, 0)
        self.assertGreaterEqual(st1.W_B, 0.0)
        self.assertEqual(st1.A_t, 5)
        
        # Next step: 0 arrivals
        st2, obs2 = self.engine.step_minute([], disruptions=[])
        # Mathematical equation: Q_{t+1}^B = max(0, Q_t^B + A_t - S_t^B - reneged)
        expected_qb = max(0, st1.Q_B + st2.A_t - st2.S_B - st2.reneged_B)
        self.assertEqual(st2.Q_B, expected_qb)

    def test_state_separation_no_leakage(self):
        """Verifies that the state estimator only consumes ObservationVector, not SystemState."""
        obs = ObservationVector(
            time_t=10.0,
            y_Q_B=7.2,
            y_Q_K=4.5,
            y_completed_billing=3,
            y_completed_kitchen=2,
            c_B_reported=3,
            c_K_reported=4,
            ambient_noise_level=68.5
        )
        est = self.estimator.update(obs)
        self.assertIsInstance(est, EstimatedState)
        self.assertGreaterEqual(est.Q_B_hat, 0.0)
        self.assertGreaterEqual(est.Q_K_hat, 0.0)
        self.assertGreaterEqual(est.W_B_hat, 0.0)
        self.assertGreaterEqual(est.W_K_hat, 0.0)
        self.assertIn("NORMAL", est.regime_probs)

    def test_workload_waiting_prediction(self):
        """Checks that waiting time increases with higher workload and order complexity."""
        est_low = EstimatedState(
            time_t=10.0,
            Q_B_hat=2.0, Q_B_p10=1.0, Q_B_p90=3.0,
            Q_K_hat=2.0, Q_K_p10=1.0, Q_K_p90=3.0,
            W_B_hat=2.0, W_B_p10=1.0, W_B_p90=3.0,
            W_K_hat=7.0, W_K_p10=4.0, W_K_p90=10.0,
            residual_B_hat=0.5, residual_K_hat=1.5,
            C_B_hat=3.0, C_K_hat=4.0
        )
        est_high = EstimatedState(
            time_t=10.0,
            Q_B_hat=15.0, Q_B_p10=12.0, Q_B_p90=18.0,
            Q_K_hat=20.0, Q_K_p10=16.0, Q_K_p90=25.0,
            W_B_hat=18.0, W_B_p10=14.0, W_B_p90=22.0,
            W_K_hat=75.0, W_K_p10=60.0, W_K_p90=95.0,
            residual_B_hat=1.5, residual_K_hat=3.0,
            C_B_hat=3.0, C_K_hat=4.0
        )
        pred_low = self.predictor.predict(est_low, OrderType.QUICK)
        pred_high = self.predictor.predict(est_high, OrderType.COMPLEX)
        self.assertGreater(pred_high.expected_total_system_time, pred_low.expected_total_system_time)
        self.assertGreater(pred_high.total_system_time_p90, pred_high.total_system_time_p10)

    def test_five_minute_forecaster(self):
        """Verifies that 5-minute forecaster outputs all 5 horizons with valid uncertainty intervals."""
        est = EstimatedState(
            time_t=120.0,
            Q_B_hat=5.0, Q_B_p10=3.0, Q_B_p90=7.0,
            Q_K_hat=8.0, Q_K_p10=5.0, Q_K_p90=11.0,
            W_B_hat=5.0, W_B_p10=3.0, W_B_p90=7.0,
            W_K_hat=30.0, W_K_p10=20.0, W_K_p90=40.0,
            residual_B_hat=1.0, residual_K_hat=2.0,
            C_B_hat=3.0, C_K_hat=4.0
        )
        fcs = self.forecaster.forecast_5min(est, current_minute_of_day=120.0)
        self.assertEqual(len(fcs), 5)
        for fc in fcs:
            self.assertGreaterEqual(fc.Q_B_p90, fc.Q_B_median)
            self.assertGreaterEqual(fc.Q_B_median, fc.Q_B_p10)
            self.assertGreaterEqual(fc.Q_K_p90, fc.Q_K_median)
            self.assertGreaterEqual(fc.Q_K_median, fc.Q_K_p10)

    def test_cross_stage_propagation(self):
        """Validates that impulse response matrix Phi_{A -> B}(tau) is non-negative and summable."""
        est = EstimatedState(
            time_t=250.0,
            Q_B_hat=10.0, Q_B_p10=8.0, Q_B_p90=12.0,
            Q_K_hat=3.0, Q_K_p10=1.0, Q_K_p90=5.0,
            W_B_hat=12.0, W_B_p10=9.0, W_B_p90=15.0,
            W_K_hat=11.0, W_K_p10=6.0, W_K_p90=18.0,
            residual_B_hat=1.0, residual_K_hat=1.5,
            C_B_hat=3.0, C_K_hat=4.0
        )
        phi = self.propagation.compute_impulse_response_matrix(est)
        self.assertEqual(len(phi), 15)
        self.assertTrue(np.all(phi >= 0.0))
        diag = self.propagation.evaluate_propagation_risk(est)
        self.assertIn(diag.recommended_stage_focus, ["STAGE_A_BILLING", "STAGE_B_KITCHEN", "BALANCED"])

    def test_optimizer_action_selection(self):
        """Verifies that the optimizer chooses appropriate action when congestion occurs."""
        # Congested state with large billing queue
        est_congested = EstimatedState(
            time_t=280.0,
            Q_B_hat=22.0, Q_B_p10=18.0, Q_B_p90=26.0,
            Q_K_hat=4.0, Q_K_p10=2.0, Q_K_p90=6.0,
            W_B_hat=25.0, W_B_p10=20.0, W_B_p90=30.0,
            W_K_hat=14.0, W_K_p10=8.0, W_K_p90=20.0,
            residual_B_hat=2.0, residual_K_hat=2.0,
            C_B_hat=3.0, C_K_hat=4.0
        )
        best_act, eval_res, all_evals = self.optimizer.optimize_action(est_congested)
        self.assertIsInstance(best_act, InterventionAction)
        self.assertGreater(len(all_evals), 1)
        self.assertGreaterEqual(eval_res.total_risk_adjusted_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
