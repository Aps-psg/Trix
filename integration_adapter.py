"""
integration_adapter.py - QueueFlux Integration Adapter & Standard State Definition

Provides a standardized data contract (QueueFluxState) and adapter logic to convert
heterogeneous data outputs from Person 1 (Queue Simulator) and Person 2 (State Estimator)
into a validated, canonical format for the Intelligence Layer.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import datetime


@dataclass
class QueueFluxState:
    """
    Standard Canonical State representation for QueueFlux at time t.
    Mathematical formulation: X_t = (Q_t^B, Q_t^K, W_t^B, W_t^K, R_t, C_t, Z_t)
    """
    # Timestamp of observation
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    # Queue Lengths (Entity counts)
    billing_queue: int = 0  # Q_t^B: Customers currently waiting in billing line
    kitchen_queue: int = 0  # Q_t^K: Billed orders currently waiting in kitchen queue

    # Rates & Durations (Real-time vs Baseline)
    arrivals_last_min: float = 3.0  # A_t: Customer arrival rate (customers/minute)
    normal_arrivals_per_min: float = 3.0  # lambda_0: Normal expected arrival rate

    billing_service_time_sec: float = 30.0  # tau^B: Observed billing service time per customer (sec)
    normal_billing_service_time_sec: float = 30.0  # tau_0^B: Normal billing service time per customer (sec)

    kitchen_service_time_sec: float = 120.0  # tau^K: Observed kitchen prep time per order (sec)
    normal_kitchen_service_time_sec: float = 120.0  # tau_0^K: Normal kitchen prep time per order (sec)

    # Active System Resources (C_t)
    active_billing_counters: int = 2  # C^B: Currently open billing counters (Cap: 4.0 cust/min)
    active_kitchen_servers: int = 8   # C^K: Currently active kitchen prep staff/stations (Cap: 4.0 ord/min)

    # Resource Pool for Interventions
    available_extra_counters: int = 1  # Unopened billing counters ready for use
    available_extra_workers: int = 2   # Available floater staff ready for deployment

    # 5-Minute Queue Length Forecasts (from Person 2 Predictor)
    forecast_billing_queue: List[float] = field(default_factory=lambda: [0.0]*5)
    forecast_kitchen_queue: List[float] = field(default_factory=lambda: [0.0]*5)

    # Summary Wait Time Forecast & Data Quality/Uncertainty
    forecast_wait_minutes: float = 0.0  # Total estimated wait time for new arrival (min)
    uncertainty: float = 0.0            # Data quality penalty / noise factor [0.0 = high confidence, 1.0 = highly uncertain]

    def validate(self) -> List[str]:
        """Validates internal field consistency and returns list of warning messages."""
        warnings = []
        if self.billing_queue < 0:
            warnings.append(f"billing_queue was negative ({self.billing_queue}), resetting to 0.")
            self.billing_queue = 0
        if self.kitchen_queue < 0:
            warnings.append(f"kitchen_queue was negative ({self.kitchen_queue}), resetting to 0.")
            self.kitchen_queue = 0
        if self.arrivals_last_min < 0:
            warnings.append(f"arrivals_last_min was negative ({self.arrivals_last_min}), resetting to 0.0.")
            self.arrivals_last_min = 0.0
        if self.billing_service_time_sec <= 0:
            warnings.append(f"billing_service_time_sec invalid ({self.billing_service_time_sec}), defaulting to 30.0s.")
            self.billing_service_time_sec = 30.0
        if self.kitchen_service_time_sec <= 0:
            warnings.append(f"kitchen_service_time_sec invalid ({self.kitchen_service_time_sec}), defaulting to 120.0s.")
            self.kitchen_service_time_sec = 120.0
        if self.active_billing_counters < 1:
            warnings.append(f"active_billing_counters must be >= 1 ({self.active_billing_counters}), setting to 1.")
            self.active_billing_counters = 1
        if self.active_kitchen_servers < 1:
            warnings.append(f"active_kitchen_servers must be >= 1 ({self.active_kitchen_servers}), setting to 1.")
            self.active_kitchen_servers = 1
        if not (0.0 <= self.uncertainty <= 1.0):
            self.uncertainty = max(0.0, min(1.0, self.uncertainty))

        # Auto-compute forecast_wait_minutes if missing (0.0)
        if self.forecast_wait_minutes <= 0.0:
            self.forecast_wait_minutes = round(self.calculate_estimated_wait_time_min(), 2)

        return warnings

    def get_billing_capacity_per_min(self) -> float:
        """Returns max billing capacity in customers/minute across all active counters."""
        if self.billing_service_time_sec <= 0 or self.active_billing_counters <= 0:
            return 0.0
        return (60.0 / self.billing_service_time_sec) * self.active_billing_counters

    def get_kitchen_capacity_per_min(self) -> float:
        """Returns max kitchen prep capacity in orders/minute across all active kitchen servers."""
        if self.kitchen_service_time_sec <= 0 or self.active_kitchen_servers <= 0:
            return 0.0
        return (60.0 / self.kitchen_service_time_sec) * self.active_kitchen_servers

    def calculate_estimated_wait_time_min(self) -> float:
        """
        Estimates total waiting time in minutes for a new customer joining the system now.
        Wait = Billing Queue Delay + Kitchen Queue Delay
        W^B = (Q^B + 0.5) / (Capacity^B)
        W^K = (Q^K + 0.5) / (Capacity^K)
        """
        b_cap = self.get_billing_capacity_per_min()
        k_cap = self.get_kitchen_capacity_per_min()

        b_wait = (self.billing_queue + 0.5) / b_cap if b_cap > 0 else 999.0
        k_wait = (self.kitchen_queue + 0.5) / k_cap if k_cap > 0 else 999.0

        return b_wait + k_wait

    def to_dict(self) -> Dict[str, Any]:
        """Serializes dataclass to standard dictionary."""
        return asdict(self)


class IntegrationAdapter:
    """
    Adapter class that maps arbitrary outputs from Person 1 (Queue Simulator)
    and Person 2 (State Predictor) into the standard QueueFluxState.
    """

    @staticmethod
    def adapt_raw_input(
        person1_data: Optional[Dict[str, Any]] = None,
        person2_data: Optional[Dict[str, Any]] = None,
        custom_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[QueueFluxState, List[str]]:
        """
        Accepts heterogeneous dictionary inputs, normalizes field names, converts units,
        handles missing data gracefully, and returns a clean QueueFluxState object.
        """
        warnings = []
        merged: Dict[str, Any] = {}

        # Merge dictionaries with priority: custom_dict > person2_data > person1_data
        if person1_data:
            merged.update(person1_data)
        if person2_data:
            merged.update(person2_data)
        if custom_dict:
            merged.update(custom_dict)

        if not merged:
            warnings.append("No input data provided to adapter. Initializing with baseline default state.")
            state = QueueFluxState()
            val_warnings = state.validate()
            return state, warnings + val_warnings

        # Flexible key lookup helper
        def get_val(keys: List[str], default: Any) -> Any:
            for k in keys:
                if k in merged and merged[k] is not None:
                    return merged[k]
            return default

        # Map Timestamp
        ts = get_val(["timestamp", "time", "clock", "event_time"], datetime.datetime.now().isoformat())

        # Map Queue Lengths
        b_q = int(get_val(["billing_queue", "q_billing", "billing_queue_length", "q_b", "billing_len"], 0))
        k_q = int(get_val(["kitchen_queue", "q_kitchen", "kitchen_queue_length", "q_k", "kitchen_len"], 0))

        # Map Arrival Rates
        arr_last = float(get_val(["arrivals_last_min", "arrival_rate", "lambda_t", "recent_arrivals", "arr_per_min"], 5.0))
        arr_norm = float(get_val(["normal_arrivals_per_min", "baseline_arrival_rate", "normal_arrivals", "lambda_0"], 5.0))

        # Map Service Durations (detect if passed in minutes vs seconds)
        raw_b_sec = get_val(["billing_service_time_sec", "billing_service_time", "tau_b", "billing_sec"], 30.0)
        b_sec = float(raw_b_sec) * 60.0 if float(raw_b_sec) < 1.0 else float(raw_b_sec)

        raw_b_norm_sec = get_val(["normal_billing_service_time_sec", "normal_billing_sec", "tau_0_b"], 30.0)
        b_norm_sec = float(raw_b_norm_sec) * 60.0 if float(raw_b_norm_sec) < 1.0 else float(raw_b_norm_sec)

        raw_k_sec = get_val(["kitchen_service_time_sec", "kitchen_service_time", "tau_k", "kitchen_sec"], 120.0)
        k_sec = float(raw_k_sec) * 60.0 if float(raw_k_sec) < 1.0 else float(raw_k_sec)

        raw_k_norm_sec = get_val(["normal_kitchen_service_time_sec", "normal_kitchen_sec", "tau_0_k"], 120.0)
        k_norm_sec = float(raw_k_norm_sec) * 60.0 if float(raw_k_norm_sec) < 1.0 else float(raw_k_norm_sec)

        # Map Resource Counts
        act_b_counters = int(get_val(["active_billing_counters", "open_counters", "c_b", "billing_counters"], 2))
        act_k_servers = int(get_val(["active_kitchen_servers", "kitchen_staff", "c_k", "kitchen_servers"], 3))

        avail_extra_c = int(get_val(["available_extra_counters", "extra_counters", "spare_counters"], 1))
        avail_extra_w = int(get_val(["available_extra_workers", "extra_workers", "spare_workers", "free_staff"], 2))

        # Map Forecast Lists
        fc_b = get_val(["forecast_billing_queue", "pred_billing_q", "b_forecast"], [float(b_q)] * 5)
        if not isinstance(fc_b, list):
            fc_b = [float(fc_b)] * 5
        fc_b = [float(x) for x in fc_b[:5]] + [0.0] * max(0, 5 - len(fc_b))

        fc_k = get_val(["forecast_kitchen_queue", "pred_kitchen_q", "k_forecast"], [float(k_q)] * 5)
        if not isinstance(fc_k, list):
            fc_k = [float(fc_k)] * 5
        fc_k = [float(x) for x in fc_k[:5]] + [0.0] * max(0, 5 - len(fc_k))

        # Map Forecast Wait Time & Uncertainty
        fc_wait = float(get_val(["forecast_wait_minutes", "predicted_wait_min", "est_wait_time"], 0.0))
        uncertainty_val = float(get_val(["uncertainty", "confidence_penalty", "noise_factor"], 0.0))

        # Build QueueFluxState object
        state = QueueFluxState(
            timestamp=str(ts),
            billing_queue=b_q,
            kitchen_queue=k_q,
            arrivals_last_min=arr_last,
            normal_arrivals_per_min=arr_norm,
            billing_service_time_sec=b_sec,
            normal_billing_service_time_sec=b_norm_sec,
            kitchen_service_time_sec=k_sec,
            normal_kitchen_service_time_sec=k_norm_sec,
            active_billing_counters=act_b_counters,
            active_kitchen_servers=act_k_servers,
            available_extra_counters=avail_extra_c,
            available_extra_workers=avail_extra_w,
            forecast_billing_queue=fc_b,
            forecast_kitchen_queue=fc_k,
            forecast_wait_minutes=fc_wait,
            uncertainty=uncertainty_val
        )

        validation_warnings = state.validate()
        warnings.extend(validation_warnings)

        return state, warnings
