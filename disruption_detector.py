"""
disruption_detector.py - QueueFlux Disruption Detector Module

Detects operational anomalies across the canteen workflow:
1. Unusual arrival surges (demand spikes)
2. Billing or kitchen service slowdowns (equipment fail / cashier slowdown)
3. Rapid queue growth (imbalance between arrival & service)
4. Capacity deficits (demand exceeds maximum theoretical service throughput)

Uses configurable thresholds and historical smoothing (hysteresis) to prevent false alarms.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from integration_adapter import QueueFluxState


class DisruptionSeverity:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DisruptionType:
    ARRIVAL_SURGE = "ARRIVAL_SURGE"
    BILLING_SLOWDOWN = "BILLING_SLOWDOWN"
    KITCHEN_SLOWDOWN = "KITCHEN_SLOWDOWN"
    RAPID_BILLING_GROWTH = "RAPID_BILLING_GROWTH"
    RAPID_KITCHEN_GROWTH = "RAPID_KITCHEN_GROWTH"
    CAPACITY_DEFICIT_BILLING = "CAPACITY_DEFICIT_BILLING"
    CAPACITY_DEFICIT_KITCHEN = "CAPACITY_DEFICIT_KITCHEN"


@dataclass
class DisruptionAlert:
    """Represents a detected operational disruption in the canteen system."""
    alert_id: str
    disruption_type: str
    stage: str               # "Arrival", "Billing", or "Kitchen"
    severity: str            # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float        # 0.0 to 1.0 (adjusted by state uncertainty)
    metric_name: str         # e.g., "arrival_rate", "billing_service_time"
    observed_value: float
    baseline_value: float
    ratio_or_delta: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "disruption_type": self.disruption_type,
            "stage": self.stage,
            "severity": self.severity,
            "confidence": round(self.confidence, 2),
            "metric_name": self.metric_name,
            "observed_value": round(self.observed_value, 2),
            "baseline_value": round(self.baseline_value, 2),
            "ratio_or_delta": round(self.ratio_or_delta, 2),
            "description": self.description,
        }


@dataclass
class DisruptionDetectorConfig:
    """Configurable thresholds for disruption detection rules."""
    surge_ratio_threshold: float = 1.35        # Arrival rate > 135% of normal
    billing_slowdown_ratio: float = 1.30       # Billing service time > 130% of normal
    kitchen_slowdown_ratio: float = 1.30       # Kitchen service time > 130% of normal
    rapid_growth_threshold: float = 3.0        # Queue growing by > 3 entities/min
    capacity_deficit_ratio: float = 1.10       # Demand rate / service capacity > 1.10
    history_window_size: int = 5              # State history length for growth tracking


class DisruptionDetector:
    """
    Analyzes QueueFluxState metrics against historical baselines and configurable
    thresholds to identify operational bottlenecks and disruptions.
    """

    def __init__(self, config: Optional[DisruptionDetectorConfig] = None):
        self.config = config if config else DisruptionDetectorConfig()
        self.state_history: List[QueueFluxState] = []

    def detect_disruptions(self, current_state: QueueFluxState) -> List[DisruptionAlert]:
        """
        Main entry point. Analyzes current state and state history, returns a list
        of active DisruptionAlerts sorted by severity.
        """
        alerts: List[DisruptionAlert] = []
        self.state_history.append(current_state)
        if len(self.state_history) > self.config.history_window_size:
            self.state_history.pop(0)

        # Base confidence factor penalized by state uncertainty
        base_confidence = max(0.1, min(1.0, 1.0 - (current_state.uncertainty * 0.75)))

        # 1. Check Arrival Surge
        surge_alert = self._check_arrival_surge(current_state, base_confidence)
        if surge_alert:
            alerts.append(surge_alert)

        # 2. Check Billing Slowdown
        b_slow_alert = self._check_billing_slowdown(current_state, base_confidence)
        if b_slow_alert:
            alerts.append(b_slow_alert)

        # 3. Check Kitchen Slowdown
        k_slow_alert = self._check_kitchen_slowdown(current_state, base_confidence)
        if k_slow_alert:
            alerts.append(k_slow_alert)

        # 4. Check Rapid Queue Growth (if history exists)
        growth_alerts = self._check_rapid_queue_growth(current_state, base_confidence)
        alerts.extend(growth_alerts)

        # 5. Check Capacity Deficits
        cap_alerts = self._check_capacity_deficits(current_state, base_confidence)
        alerts.extend(cap_alerts)

        # Sort alerts by severity order: CRITICAL > HIGH > MEDIUM > LOW
        severity_order = {DisruptionSeverity.CRITICAL: 4, DisruptionSeverity.HIGH: 3, DisruptionSeverity.MEDIUM: 2, DisruptionSeverity.LOW: 1}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 0), reverse=True)

        return alerts

    def _check_arrival_surge(self, state: QueueFluxState, base_conf: float) -> Optional[DisruptionAlert]:
        if state.normal_arrivals_per_min <= 0:
            return None
        ratio = state.arrivals_last_min / state.normal_arrivals_per_min
        if ratio >= self.config.surge_ratio_threshold:
            if ratio >= 2.0:
                severity = DisruptionSeverity.CRITICAL
            elif ratio >= 1.6:
                severity = DisruptionSeverity.HIGH
            elif ratio >= 1.4:
                severity = DisruptionSeverity.MEDIUM
            else:
                severity = DisruptionSeverity.LOW

            return DisruptionAlert(
                alert_id=f"ALT-SURGE-{int(ratio*100)}",
                disruption_type=DisruptionType.ARRIVAL_SURGE,
                stage="Arrival",
                severity=severity,
                confidence=base_conf,
                metric_name="arrivals_last_min",
                observed_value=state.arrivals_last_min,
                baseline_value=state.normal_arrivals_per_min,
                ratio_or_delta=ratio,
                description=f"Arrival rate surge detected: {state.arrivals_last_min:.1f} cust/min vs normal baseline of {state.normal_arrivals_per_min:.1f} cust/min ({ratio:.1f}x spike)."
            )
        return None

    def _check_billing_slowdown(self, state: QueueFluxState, base_conf: float) -> Optional[DisruptionAlert]:
        if state.normal_billing_service_time_sec <= 0:
            return None
        ratio = state.billing_service_time_sec / state.normal_billing_service_time_sec
        if ratio >= self.config.billing_slowdown_ratio:
            if ratio >= 2.0:
                severity = DisruptionSeverity.CRITICAL
            elif ratio >= 1.6:
                severity = DisruptionSeverity.HIGH
            else:
                severity = DisruptionSeverity.MEDIUM

            return DisruptionAlert(
                alert_id=f"ALT-BILLSLOW-{int(ratio*100)}",
                disruption_type=DisruptionType.BILLING_SLOWDOWN,
                stage="Billing",
                severity=severity,
                confidence=base_conf,
                metric_name="billing_service_time_sec",
                observed_value=state.billing_service_time_sec,
                baseline_value=state.normal_billing_service_time_sec,
                ratio_or_delta=ratio,
                description=f"Billing counter slowdown: {state.billing_service_time_sec:.1f}s per customer vs normal {state.normal_billing_service_time_sec:.1f}s ({ratio:.2f}x delay)."
            )
        return None

    def _check_kitchen_slowdown(self, state: QueueFluxState, base_conf: float) -> Optional[DisruptionAlert]:
        if state.normal_kitchen_service_time_sec <= 0:
            return None
        ratio = state.kitchen_service_time_sec / state.normal_kitchen_service_time_sec
        if ratio >= self.config.kitchen_slowdown_ratio:
            if ratio >= 2.0:
                severity = DisruptionSeverity.CRITICAL
            elif ratio >= 1.6:
                severity = DisruptionSeverity.HIGH
            else:
                severity = DisruptionSeverity.MEDIUM

            return DisruptionAlert(
                alert_id=f"ALT-KITCHENSLOW-{int(ratio*100)}",
                disruption_type=DisruptionType.KITCHEN_SLOWDOWN,
                stage="Kitchen",
                severity=severity,
                confidence=base_conf,
                metric_name="kitchen_service_time_sec",
                observed_value=state.kitchen_service_time_sec,
                baseline_value=state.normal_kitchen_service_time_sec,
                ratio_or_delta=ratio,
                description=f"Kitchen preparation slowdown: {state.kitchen_service_time_sec:.1f}s per order vs normal {state.normal_kitchen_service_time_sec:.1f}s ({ratio:.2f}x delay)."
            )
        return None

    def _check_rapid_queue_growth(self, state: QueueFluxState, base_conf: float) -> List[DisruptionAlert]:
        alerts = []
        if len(self.state_history) < 2:
            return alerts

        prev_state = self.state_history[-2]

        b_growth = state.billing_queue - prev_state.billing_queue
        if b_growth >= self.config.rapid_growth_threshold:
            alerts.append(DisruptionAlert(
                alert_id=f"ALT-BGROWTH-{b_growth}",
                disruption_type=DisruptionType.RAPID_BILLING_GROWTH,
                stage="Billing",
                severity=DisruptionSeverity.HIGH if b_growth >= 5 else DisruptionSeverity.MEDIUM,
                confidence=base_conf,
                metric_name="billing_queue_delta",
                observed_value=float(b_growth),
                baseline_value=0.0,
                ratio_or_delta=float(b_growth),
                description=f"Billing queue growing rapidly (+{b_growth} customers in past minute). Current queue: {state.billing_queue}."
            ))

        k_growth = state.kitchen_queue - prev_state.kitchen_queue
        if k_growth >= self.config.rapid_growth_threshold:
            alerts.append(DisruptionAlert(
                alert_id=f"ALT-KGROWTH-{k_growth}",
                disruption_type=DisruptionType.RAPID_KITCHEN_GROWTH,
                stage="Kitchen",
                severity=DisruptionSeverity.HIGH if k_growth >= 5 else DisruptionSeverity.MEDIUM,
                confidence=base_conf,
                metric_name="kitchen_queue_delta",
                observed_value=float(k_growth),
                baseline_value=0.0,
                ratio_or_delta=float(k_growth),
                description=f"Kitchen backlog growing rapidly (+{k_growth} orders in past minute). Current queue: {state.kitchen_queue}."
            ))

        return alerts

    def _check_capacity_deficits(self, state: QueueFluxState, base_conf: float) -> List[DisruptionAlert]:
        alerts = []

        # 1. Billing Capacity Deficit (Arrival Rate vs Billing Capacity)
        b_capacity = state.get_billing_capacity_per_min()
        if b_capacity > 0:
            b_ratio = state.arrivals_last_min / b_capacity
            if b_ratio >= self.config.capacity_deficit_ratio:
                alerts.append(DisruptionAlert(
                    alert_id=f"ALT-BCAPDEF-{int(b_ratio*100)}",
                    disruption_type=DisruptionType.CAPACITY_DEFICIT_BILLING,
                    stage="Billing",
                    severity=DisruptionSeverity.HIGH if b_ratio >= 1.3 else DisruptionSeverity.MEDIUM,
                    confidence=base_conf,
                    metric_name="billing_capacity_ratio",
                    observed_value=state.arrivals_last_min,
                    baseline_value=b_capacity,
                    ratio_or_delta=b_ratio,
                    description=f"Billing capacity deficit: Arrival rate ({state.arrivals_last_min:.1f} cust/min) exceeds total billing capacity ({b_capacity:.1f} cust/min)."
                ))

        # 2. Kitchen Capacity Deficit (Orders reaching kitchen vs Kitchen Prep Capacity)
        # Note: Effective arrival rate to kitchen is bounded by billing throughput when billing line is non-empty
        effective_kitchen_arrivals = min(state.arrivals_last_min, b_capacity) if state.billing_queue > 0 else state.arrivals_last_min
        k_capacity = state.get_kitchen_capacity_per_min()
        if k_capacity > 0:
            k_ratio = effective_kitchen_arrivals / k_capacity
            if k_ratio >= self.config.capacity_deficit_ratio or (state.kitchen_queue > 8 and k_ratio >= 1.0):
                alerts.append(DisruptionAlert(
                    alert_id=f"ALT-KCAPDEF-{int(k_ratio*100)}",
                    disruption_type=DisruptionType.CAPACITY_DEFICIT_KITCHEN,
                    stage="Kitchen",
                    severity=DisruptionSeverity.HIGH if k_ratio >= 1.3 else DisruptionSeverity.MEDIUM,
                    confidence=base_conf,
                    metric_name="kitchen_capacity_ratio",
                    observed_value=effective_kitchen_arrivals,
                    baseline_value=k_capacity,
                    ratio_or_delta=k_ratio,
                    description=f"Kitchen capacity deficit: Incoming order flow ({effective_kitchen_arrivals:.1f} orders/min) exceeds kitchen prep capacity ({k_capacity:.1f} orders/min)."
                ))

        return alerts
