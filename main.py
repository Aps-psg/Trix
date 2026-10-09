"""
main.py - QueueFlux End-to-End Demonstration Script

Executes 6 realistic canteen operational scenarios to demonstrate the QueueFlux
Intelligence Layer in action:
1. Normal Canteen Operation
2. Sudden Arrival Surge
3. Billing Counter Slowdown
4. Kitchen Preparation Slowdown
5. Severely Resource-Constrained Operational Surge
6. High Data Uncertainty & Incomplete Forecasts
"""

import sys
import json
from intelligence_engine import IntelligenceEngine
from integration_adapter import QueueFluxState


def run_scenario(scenario_num: int, scenario_title: str, input_data: dict):
    print("\n" + "#" * 80)
    print(f" SCENARIO {scenario_num}: {scenario_title.upper()}")
    print("#" * 80)

    engine = IntelligenceEngine()
    report = engine.process_raw_input(custom_dict=input_data)

    # Output Human-Readable Summary
    print(report.to_human_summary())

    # Output Machine-Readable JSON sample snippet
    json_data = report.to_dict()
    print("\n[MACHINE-READABLE DASHBOARD PAYLOAD SNIPPET (JSON)]")
    snippet = {
        "timestamp": json_data["timestamp"],
        "disruptions_count": len(json_data["detected_disruptions"]),
        "recommended_action": json_data["recommended_action"]["action_name"],
        "target_stage": json_data["recommended_action"]["target_stage"],
        "baseline_wait_min": json_data["impact_comparison"]["baseline_wait_minutes"],
        "expected_wait_min": json_data["impact_comparison"]["expected_wait_minutes"],
        "savings_min": json_data["impact_comparison"]["estimated_wait_reduction_minutes"],
    }
    print(json.dumps(snippet, indent=2))


def main():
    print("=" * 80)
    print("        QUEUEFLUX: INTELLIGENT CANTEEN QUEUE MANAGEMENT SYSTEM")
    print("              Demonstrating Intelligence Layer & Integration")
    print("=" * 80)

    # SCENARIO 1: Normal Operation
    normal_input = {
        "timestamp": "2026-10-09T12:00:00",
        "billing_queue": 2,
        "kitchen_queue": 3,
        "arrivals_last_min": 5.0,
        "normal_arrivals_per_min": 5.0,
        "billing_service_time_sec": 30.0,
        "normal_billing_service_time_sec": 30.0,
        "kitchen_service_time_sec": 120.0,
        "normal_kitchen_service_time_sec": 120.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 3,
        "available_extra_counters": 1,
        "available_extra_workers": 2,
        "uncertainty": 0.05
    }
    run_scenario(1, "Normal Canteen Operation", normal_input)

    # SCENARIO 2: Sudden Arrival Surge
    surge_input = {
        "timestamp": "2026-10-09T12:15:00",
        "billing_queue": 14,
        "kitchen_queue": 5,
        "arrivals_last_min": 14.0,
        "normal_arrivals_per_min": 5.0,
        "billing_service_time_sec": 30.0,
        "normal_billing_service_time_sec": 30.0,
        "kitchen_service_time_sec": 120.0,
        "normal_kitchen_service_time_sec": 120.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 3,
        "available_extra_counters": 1,
        "available_extra_workers": 2,
        "forecast_billing_queue": [14.0, 16.0, 18.0, 15.0, 12.0],
        "forecast_kitchen_queue": [5.0, 8.0, 12.0, 14.0, 10.0],
        "uncertainty": 0.10
    }
    run_scenario(2, "Sudden Lunch Arrival Surge", surge_input)

    # SCENARIO 3: Billing Slowdown
    billing_slowdown_input = {
        "timestamp": "2026-10-09T12:30:00",
        "billing_queue": 16,
        "kitchen_queue": 2,
        "arrivals_last_min": 6.0,
        "normal_arrivals_per_min": 5.0,
        "billing_service_time_sec": 75.0,   # Slowdown: 75s vs 30s normal
        "normal_billing_service_time_sec": 30.0,
        "kitchen_service_time_sec": 120.0,
        "normal_kitchen_service_time_sec": 120.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 3,
        "available_extra_counters": 1,
        "available_extra_workers": 2,
        "uncertainty": 0.10
    }
    run_scenario(3, "Billing Scanner / Cashier Slowdown", billing_slowdown_input)

    # SCENARIO 4: Kitchen Slowdown
    kitchen_slowdown_input = {
        "timestamp": "2026-10-09T12:45:00",
        "billing_queue": 3,
        "kitchen_queue": 16,
        "arrivals_last_min": 5.0,
        "normal_arrivals_per_min": 5.0,
        "billing_service_time_sec": 30.0,
        "normal_billing_service_time_sec": 30.0,
        "kitchen_service_time_sec": 240.0,  # Slowdown: 240s vs 120s normal
        "normal_kitchen_service_time_sec": 120.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 3,
        "available_extra_counters": 1,
        "available_extra_workers": 2,
        "uncertainty": 0.15
    }
    run_scenario(4, "Kitchen Preparation Station Slowdown", kitchen_slowdown_input)

    # SCENARIO 5: Resource-Constrained Surge (0 extra staff/counters available)
    constrained_input = {
        "timestamp": "2026-10-09T13:00:00",
        "billing_queue": 15,
        "kitchen_queue": 4,
        "arrivals_last_min": 12.0,
        "normal_arrivals_per_min": 5.0,
        "billing_service_time_sec": 30.0,
        "normal_billing_service_time_sec": 30.0,
        "kitchen_service_time_sec": 120.0,
        "normal_kitchen_service_time_sec": 120.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 3,
        "available_extra_counters": 0,     # NO extra counters
        "available_extra_workers": 0,      # NO extra workers available
        "uncertainty": 0.10
    }
    run_scenario(5, "Resource-Constrained Surge (0 Extra Staff/Counters)", constrained_input)

    # SCENARIO 6: High Data Uncertainty & Incomplete Teammate Data
    uncertain_input = {
        "time": "2026-10-09T13:15:00",
        "q_b": 12,
        # Missing q_k (kitchen queue), will default safely
        "arr_rate": 9.0,
        "billing_sec": 45.0,
        # Missing kitchen service time
        "open_counters": 2,
        "spare_counters": 1,
        "spare_workers": 1,
        "uncertainty": 0.65  # High data noise / uncertainty factor
    }
    run_scenario(6, "High Uncertainty & Incomplete Forecast Data", uncertain_input)

    print("\n" + "=" * 80)
    print("       ALL 6 DEMONSTRATION SCENARIOS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
