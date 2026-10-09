"""
test_intelligence.py - QueueFlux Unit & Integration Test Suite

Comprehensive test coverage for all QueueFlux Intelligence modules using unittest.
Tests input adaptation, disruption detection, cross-stage propagation analysis,
decision optimization, evaluation metrics, and end-to-end integration.
"""

import unittest
from integration_adapter import QueueFluxState, IntegrationAdapter
from disruption_detector import DisruptionDetector, DisruptionDetectorConfig, DisruptionType, DisruptionSeverity
from propagation_analyzer import PropagationAnalyzer
from decision_optimizer import DecisionOptimizer, ActionType
from evaluation import EvaluationEngine
from intelligence_engine import IntelligenceEngine


class TestIntegrationAdapter(unittest.TestCase):

    def test_valid_input_adaptation(self):
        p1 = {"billing_queue": 10, "kitchen_queue": 4, "arrivals_last_min": 12.0}
        p2 = {"active_billing_counters": 2, "active_kitchen_servers": 3, "forecast_wait_minutes": 8.5}

        state, warnings = IntegrationAdapter.adapt_raw_input(person1_data=p1, person2_data=p2)
        self.assertEqual(state.billing_queue, 10)
        self.assertEqual(state.kitchen_queue, 4)
        self.assertEqual(state.arrivals_last_min, 12.0)
        self.assertEqual(state.active_billing_counters, 2)
        self.assertEqual(state.forecast_wait_minutes, 8.5)

    def test_missing_fields_and_fallbacks(self):
        # Empty inputs should fall back gracefully with default state and warnings
        state, warnings = IntegrationAdapter.adapt_raw_input(custom_dict={})
        self.assertGreater(len(warnings), 0)
        self.assertEqual(state.billing_queue, 0)
        self.assertGreater(state.forecast_wait_minutes, 0.0)

    def test_unit_conversion_and_key_mapping(self):
        raw = {
            "q_b": 8,
            "billing_sec": 40.0,
            "extra_workers": 3
        }
        state, warnings = IntegrationAdapter.adapt_raw_input(custom_dict=raw)
        self.assertEqual(state.billing_queue, 8)
        self.assertEqual(state.billing_service_time_sec, 40.0)
        self.assertEqual(state.available_extra_workers, 3)


class TestDisruptionDetector(unittest.TestCase):

    def setUp(self):
        self.detector = DisruptionDetector()

    def test_normal_state_no_disruptions(self):
        normal_state = QueueFluxState(
            arrivals_last_min=3.0,
            normal_arrivals_per_min=3.0,
            billing_service_time_sec=30.0,
            normal_billing_service_time_sec=30.0,
            kitchen_service_time_sec=120.0,
            normal_kitchen_service_time_sec=120.0,
            active_billing_counters=2,
            active_kitchen_servers=8,
            billing_queue=0,
            kitchen_queue=0
        )
        alerts = self.detector.detect_disruptions(normal_state)
        self.assertEqual(len(alerts), 0)

    def test_arrival_surge_detection(self):
        surge_state = QueueFluxState(
            arrivals_last_min=15.0,  # 3x normal
            normal_arrivals_per_min=5.0
        )
        alerts = self.detector.detect_disruptions(surge_state)
        surge_alerts = [a for a in alerts if a.disruption_type == DisruptionType.ARRIVAL_SURGE]
        self.assertEqual(len(surge_alerts), 1)
        self.assertEqual(surge_alerts[0].severity, DisruptionSeverity.CRITICAL)

    def test_billing_slowdown_detection(self):
        slow_state = QueueFluxState(
            billing_service_time_sec=75.0,  # 2.5x normal
            normal_billing_service_time_sec=30.0
        )
        alerts = self.detector.detect_disruptions(slow_state)
        slow_alerts = [a for a in alerts if a.disruption_type == DisruptionType.BILLING_SLOWDOWN]
        self.assertEqual(len(slow_alerts), 1)

    def test_kitchen_slowdown_detection(self):
        slow_state = QueueFluxState(
            kitchen_service_time_sec=240.0,  # 2.0x normal
            normal_kitchen_service_time_sec=120.0
        )
        alerts = self.detector.detect_disruptions(slow_state)
        slow_alerts = [a for a in alerts if a.disruption_type == DisruptionType.KITCHEN_SLOWDOWN]
        self.assertEqual(len(slow_alerts), 1)


class TestPropagationAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = PropagationAnalyzer()

    def test_billing_slowdown_propagation(self):
        detector = DisruptionDetector()
        state = QueueFluxState(
            billing_queue=15,
            billing_service_time_sec=75.0,
            normal_billing_service_time_sec=30.0
        )
        alerts = detector.detect_disruptions(state)
        effects = self.analyzer.analyze_propagation(state, alerts)

        self.assertGreater(len(effects), 0)
        target_stages = {e.target_stage for e in effects}
        self.assertIn("Kitchen Queue", target_stages)
        self.assertIn("Food Collection", target_stages)


class TestDecisionOptimizer(unittest.TestCase):

    def setUp(self):
        self.optimizer = DecisionOptimizer()
        self.detector = DisruptionDetector()
        self.propagation_analyzer = PropagationAnalyzer()

    def test_open_extra_counter_recommendation(self):
        surge_state = QueueFluxState(
            billing_queue=14,
            arrivals_last_min=14.0,
            normal_arrivals_per_min=5.0,
            available_extra_counters=1,
            available_extra_workers=2
        )
        alerts = self.detector.detect_disruptions(surge_state)
        effects = self.propagation_analyzer.analyze_propagation(surge_state, alerts)

        best_action, candidates = self.optimizer.optimize_decision(surge_state, alerts, effects)
        self.assertEqual(best_action.action_name, ActionType.OPEN_EXTRA_BILLING_COUNTER)
        self.assertTrue(best_action.is_feasible)

    def test_resource_constraint_enforcement(self):
        # Surge occurs but 0 extra workers available
        constrained_state = QueueFluxState(
            billing_queue=14,
            arrivals_last_min=14.0,
            available_extra_counters=1,
            available_extra_workers=0  # INSUFFICIENT WORKERS
        )
        alerts = self.detector.detect_disruptions(constrained_state)
        effects = self.propagation_analyzer.analyze_propagation(constrained_state, alerts)

        best_action, candidates = self.optimizer.optimize_decision(constrained_state, alerts, effects)

        # OPEN_EXTRA_BILLING_COUNTER must be marked infeasible
        open_counter_candidate = next(c for c in candidates if c.action_name == ActionType.OPEN_EXTRA_BILLING_COUNTER)
        self.assertFalse(open_counter_candidate.is_feasible)
        self.assertIn("Insufficient extra staff", open_counter_candidate.constraint_notes)

    def test_kitchen_worker_allocation(self):
        kitchen_state = QueueFluxState(
            kitchen_queue=16,
            kitchen_service_time_sec=240.0,
            normal_kitchen_service_time_sec=120.0,
            available_extra_workers=1
        )
        alerts = self.detector.detect_disruptions(kitchen_state)
        effects = self.propagation_analyzer.analyze_propagation(kitchen_state, alerts)

        best_action, candidates = self.optimizer.optimize_decision(kitchen_state, alerts, effects)
        self.assertEqual(best_action.action_name, ActionType.ALLOCATE_WORKER_TO_KITCHEN)


class TestEvaluationEngine(unittest.TestCase):

    def test_disruption_detection_evaluation(self):
        y_true = [True, True, False, False, True]
        y_pred = [True, False, False, False, True]
        delays = [15.0, 30.0]

        report = EvaluationEngine.evaluate_disruption_detection(y_true, y_pred, delays)
        self.assertEqual(report.status, "SUCCESS")
        self.assertEqual(report.true_positives, 2)
        self.assertEqual(report.false_negatives, 1)
        self.assertAlmostEqual(report.precision, 1.0)
        self.assertAlmostEqual(report.recall, 2 / 3)

    def test_waiting_time_prediction_evaluation(self):
        actuals = [5.0, 10.0, 15.0]
        preds = [4.5, 11.0, 14.0]

        report = EvaluationEngine.evaluate_waiting_time_predictions(actuals, preds)
        self.assertEqual(report.status, "SUCCESS")
        self.assertAlmostEqual(report.mae_minutes, (0.5 + 1.0 + 1.0) / 3)

    def test_intervention_impact_evaluation(self):
        base_waits = [12.0, 14.0, 16.0]
        inter_waits = [6.0, 7.0, 8.0]

        report = EvaluationEngine.evaluate_intervention_impact(base_waits, inter_waits)
        self.assertEqual(report.status, "SUCCESS")
        self.assertEqual(report.baseline_avg_wait_min, 14.0)
        self.assertEqual(report.intervention_avg_wait_min, 7.0)
        self.assertEqual(report.percentage_reduction, 50.0)

    def test_missing_metrics_handling(self):
        report = EvaluationEngine.evaluate_disruption_detection([], [])
        self.assertTrue("UNAVAILABLE" in report.status)


class TestIntelligenceEngineEndToEnd(unittest.TestCase):

    def test_end_to_end_pipeline(self):
        engine = IntelligenceEngine()
        raw_input = {
            "billing_queue": 12,
            "kitchen_queue": 4,
            "arrivals_last_min": 15.0,
            "available_extra_counters": 1,
            "available_extra_workers": 2
        }

        report = engine.process_raw_input(custom_dict=raw_input)
        self.assertIsNotNone(report.timestamp)
        self.assertGreater(len(report.detected_disruptions), 0)
        self.assertIsNotNone(report.recommended_action)
        self.assertIsNotNone(report.to_dict())
        self.assertIn("QUEUEFLUX INTELLIGENCE REPORT", report.to_human_summary())


if __name__ == "__main__":
    unittest.main()
