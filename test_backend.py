"""
test_backend.py - Integration Tests for QueueFlux FastAPI Backend API

Tests all REST API endpoints, state normalization, decision optimization pipeline,
preset operational scenarios, simulation state control, and error handling.
Authored by: sbsureka (https://github.com/sbsureka)
"""

import unittest
import json
from fastapi.testclient import TestClient
from server import app, sim_manager


class TestQueueFluxBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        sim_manager.reset_simulation(scenario_name="normal")

    def test_01_root_and_health_check(self):
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertEqual(res_root.json()["app"], "QueueFlux Backend Engine API")

        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "healthy")

    def test_02_get_scenarios(self):
        res = self.client.get("/api/scenarios")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("scenarios", data)
        self.assertIn("normal", data["scenarios"])
        self.assertIn("arrival_surge", data["scenarios"])
        self.assertIn("billing_slowdown", data["scenarios"])
        self.assertIn("kitchen_slowdown", data["scenarios"])

    def test_03_run_preset_scenario(self):
        payload = {"scenario": "arrival_surge"}
        res = self.client.post("/api/scenarios/run", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["scenario"], "arrival_surge")
        self.assertIn("report", data)
        report = data["report"]
        self.assertIn("detected_disruptions", report)
        self.assertIn("recommended_action", report)

    def test_04_simulation_state_and_step(self):
        res_state = self.client.get("/api/simulation/state")
        self.assertEqual(res_state.status_code, 200)
        state_data = res_state.json()
        self.assertIn("estimated_data", state_data)

        res_step = self.client.post("/api/simulation/step", json={"duration_minutes": 2})
        self.assertEqual(res_step.status_code, 200)

    def test_05_simulation_reset(self):
        res = self.client.post("/api/simulation/reset", json={"scenario": "kitchen_slowdown"})
        self.assertEqual(res.status_code, 200)

    def test_06_apply_manager_action(self):
        res = self.client.post("/api/simulation/apply-action", json={
            "action_name": "OPEN_EXTRA_BILLING_COUNTER",
            "target_stage": "Billing"
        })
        self.assertEqual(res.status_code, 200)

    def test_07_adapt_raw_input(self):
        custom_input = {
            "custom_dict": {
                "q_b": 10,
                "q_k": 5,
                "lambda_t": 12.0,
                "billing_sec": 40.0,
                "kitchen_sec": 120.0,
                "open_counters": 2,
                "kitchen_staff": 8
            }
        }
        res = self.client.post("/api/adapt", json=custom_input)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        normalized = data["normalized_state"]
        self.assertEqual(normalized["billing_queue"], 10)

    def test_08_evaluate_custom_state(self):
        custom_input = {
            "custom_dict": {
                "billing_queue": 15,
                "kitchen_queue": 4,
                "arrivals_last_min": 14.0,
                "normal_arrivals_per_min": 5.0,
                "billing_service_time_sec": 30.0,
                "kitchen_service_time_sec": 120.0,
                "active_billing_counters": 2,
                "active_kitchen_servers": 8,
                "available_extra_counters": 1,
                "available_extra_workers": 2
            }
        }
        res = self.client.post("/api/evaluate", json=custom_input)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        report = data["report"]
        self.assertIn("recommended_action", report)


if __name__ == "__main__":
    unittest.main()
