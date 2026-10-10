"""QueueFlux End-to-End Consumer App Flow & Billing Telemetry Test Suite
====================================================================
Tests:
1. Student Registration & UID Generation with College Roll No
2. Real-Time Billing Counter Telemetry & Mathematical Wait Estimator
3. Interactive Billing Telemetry Adjustments (+1 customer, counters)
4. Consumer Cart Wait-Time Estimation (Stage A Billing + Stage B Kitchen)
5. Order Creation & Digital Queue Token Issuance
6. Staff POS Queue Lifecycle & Order Advance (Billing -> Kitchen -> Ready)
7. Manager Multi-Agent Control Tower (Person 1 ML + Person 2 Diagnostics)
"""

import pytest
from fastapi.testclient import TestClient
from api import app, live_canteen

client = TestClient(app)


def test_student_registration_and_uid_generation():
    """Verify that registering with College Roll No generates an official STU-ROLL-xxxx UID."""
    payload = {
        "roll_no": "24CS108",
        "name": "Alex Chen",
        "department": "Computer Science & Engineering"
    }
    response = client.post("/api/auth/register-student", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Alex Chen"
    assert data["roll_no"] == "24CS108"
    assert "user_id" in data
    assert data["user_id"].startswith("STU-24CS108-")


def test_billing_telemetry_and_calculation():
    """Verify that billing counter inputs calculate the correct waiting time."""
    response = client.get("/api/billing/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "queue_length" in data
    assert "active_counters" in data
    assert "avg_service_time" in data
    assert "billing_wait_min" in data
    assert "formula" in data
    assert data["billing_wait_min"] >= 0.0


def test_billing_telemetry_update():
    """Verify that modifying billing counter inputs immediately recalculates waiting time."""
    update_payload = {
        "queue_length": 6,
        "active_counters": 2,
        "avg_service_time": 1.2
    }
    response = client.post("/api/billing/update-telemetry", json=update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["queue_length"] == 6
    assert data["active_counters"] == 2
    # Service throughput = 2 / 1.2 = 1.67 cust/min
    # Wait = (6 + 1) / 1.67 = 4.2 mins
    assert 4.0 <= data["billing_wait_min"] <= 4.5


def test_consumer_wait_estimation():
    """Verify full journey wait estimation combining Stage A Billing and Stage B Kitchen prep."""
    cart_payload = {
        "items": [
            {
                "item_id": "item_meal_1",
                "name": "Executive Lunch Thali",
                "category": "MEAL",
                "quantity": 1,
                "base_prep_time_min": 5.0,
                "price": 95.0
            }
        ]
    }
    response = client.post("/api/consumer/estimate", json=cart_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["billing_wait_min"] > 0
    assert data["kitchen_prep_min"] >= 5.0
    assert data["total_wait_min"] == pytest.approx(data["billing_wait_min"] + data["queue_delay_min"] + data["kitchen_prep_min"], 0.2)
    assert "ready_time_str" in data


def test_place_order_and_staff_lifecycle():
    """Verify order placement, token creation, and progression through billing to pickup."""
    order_payload = {
        "student_id": "STU-24CS108-9999",
        "student_name": "Alex Chen",
        "items": [
            {
                "item_id": "item_bev_1",
                "name": "Filter Coffee",
                "category": "BEVERAGE",
                "quantity": 1,
                "base_prep_time_min": 1.2,
                "price": 25.0
            }
        ]
    }
    order_res = client.post("/api/consumer/orders", json=order_payload)
    assert order_res.status_code == 200
    order_data = order_res.json()
    assert order_data["status"] == "success"
    order_id = order_data["order"]["order_id"]

    # Check staff queue has this order in WAITING_BILLING
    queue_res = client.get("/api/staff/queue")
    assert queue_res.status_code == 200
    staff_queue = queue_res.json()
    billing_order_ids = [o["order_id"] for o in staff_queue["billing_stage"]]
    assert order_id in billing_order_ids

    # Advance to PREPARING
    adv1 = client.post(f"/api/staff/orders/{order_id}/advance")
    assert adv1.status_code == 200
    assert adv1.json()["current_status"] == "PREPARING"

    # Advance to READY
    adv2 = client.post(f"/api/staff/orders/{order_id}/advance")
    assert adv2.status_code == 200
    assert adv2.json()["current_status"] == "READY"


def test_manager_control_tower_integration():
    """Verify Person 1 ML forecasts and Person 2 root-cause recommendations integration."""
    res = client.get("/api/manager/control-tower")
    assert res.status_code == 200
    data = res.json()
    assert "person1_prediction" in data
    assert "predicted_queue_length" in data["person1_prediction"]
    assert "person2_intelligence" in data
    assert "congestion_cause" in data["person2_intelligence"]
    assert "recommended_action" in data["person2_intelligence"]
