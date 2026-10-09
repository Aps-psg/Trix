"""
propagation_analyzer.py - QueueFlux Propagation Analyzer Module

Analyzes cross-stage disruption propagation in the canteen workflow:
Customer Arrival -> Billing Counter (Stage A) -> Food Prep Kitchen (Stage B) -> Food Collection

Key Architectural Insights:
1. A Billing Slowdown increases billing queue and delays order registration.
   It reduces the immediate order inflow into the kitchen (temporary starvation),
   but builds up hidden kitchen demand that will surge once billing resolves.
2. An Arrival Surge immediately swells the billing queue. If billing has high capacity,
   the surge propagates downstream to the kitchen queue within 2-4 minutes.
3. A Kitchen Slowdown causes a severe backlog in kitchen orders without affecting
   billing queue length, leading to collection delays.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from integration_adapter import QueueFluxState
from disruption_detector import DisruptionAlert, DisruptionType, DisruptionSeverity


@dataclass
class PropagationEffect:
    """Represents a predicted downstream or upstream propagation effect of a disruption."""
    source_stage: str              # "Arrival", "Billing", or "Kitchen"
    target_stage: str              # "Billing Queue", "Kitchen Queue", or "Food Collection"
    mechanism: str                 # Architectural description of propagation path
    time_lag_minutes: float        # Projected time before downstream peak impact occurs
    predicted_impact_severity: str # LOW, MEDIUM, HIGH, CRITICAL
    vulnerability_score: float     # 0.0 (resilient) to 1.0 (highly vulnerable)
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_stage": self.source_stage,
            "target_stage": self.target_stage,
            "mechanism": self.mechanism,
            "time_lag_minutes": round(self.time_lag_minutes, 1),
            "predicted_impact_severity": self.predicted_impact_severity,
            "vulnerability_score": round(self.vulnerability_score, 2),
            "description": self.description,
        }


class PropagationAnalyzer:
    """
    Evaluates stage relationships, transition rates, and current queue workloads
    to predict how disruptions cascade through the canteen lifecycle.
    """

    def analyze_propagation(
        self,
        state: QueueFluxState,
        active_disruptions: List[DisruptionAlert]
    ) -> List[PropagationEffect]:
        """
        Analyzes active disruptions and state metrics to generate cross-stage propagation forecasts.
        """
        effects: List[PropagationEffect] = []

        if not active_disruptions:
            return effects

        disruption_types = {d.disruption_type for d in active_disruptions}
        b_cap = state.get_billing_capacity_per_min()
        k_cap = state.get_kitchen_capacity_per_min()

        # 1. Billing Slowdown / Bottleneck Propagation Analysis
        if DisruptionType.BILLING_SLOWDOWN in disruption_types or DisruptionType.CAPACITY_DEFICIT_BILLING in disruption_types:
            # Calculate time lag for billing queue traversal
            time_in_billing_queue = (state.billing_queue / b_cap) if b_cap > 0 else 10.0

            # Evaluate downstream impact on kitchen: orders are delayed reaching kitchen
            effects.append(PropagationEffect(
                source_stage="Billing",
                target_stage="Kitchen Queue",
                mechanism="Order Inflow Throttling & Deferred Kitchen Surge",
                time_lag_minutes=round(time_in_billing_queue, 1),
                predicted_impact_severity=DisruptionSeverity.HIGH if state.billing_queue > 10 else DisruptionSeverity.MEDIUM,
                vulnerability_score=min(1.0, state.billing_queue / 15.0),
                description=(
                    f"Billing slowdown is holding back {state.billing_queue} customers. Kitchen is currently receiving orders "
                    f"at a throttled rate ({b_cap:.1f} orders/min). Once billing resolves, a burst of {state.billing_queue} "
                    f"orders will flood the kitchen queue with an estimated lag of {time_in_billing_queue:.1f} minutes."
                )
            ))

            effects.append(PropagationEffect(
                source_stage="Billing",
                target_stage="Food Collection",
                mechanism="Cumulative Customer Delay Cascade",
                time_lag_minutes=round(time_in_billing_queue + 2.0, 1),
                predicted_impact_severity=DisruptionSeverity.HIGH,
                vulnerability_score=0.85,
                description=f"Total end-to-end customer wait time will increase by approximately {time_in_billing_queue:.1f} minutes due to billing bottleneck."
            ))

        # 2. Arrival Surge Propagation Analysis
        if DisruptionType.ARRIVAL_SURGE in disruption_types:
            # Surge hits billing first. How long until it hits kitchen?
            # If billing capacity > arrival rate, it flows to kitchen immediately (~1-2 mins).
            # If billing capacity < arrival rate, it accumulates at billing first.
            if b_cap >= state.arrivals_last_min:
                prop_lag = 1.5
                k_severity = DisruptionSeverity.HIGH if state.arrivals_last_min > k_cap else DisruptionSeverity.MEDIUM
                effects.append(PropagationEffect(
                    source_stage="Arrival",
                    target_stage="Kitchen Queue",
                    mechanism="Fast Throughput Order Surge",
                    time_lag_minutes=prop_lag,
                    predicted_impact_severity=k_severity,
                    vulnerability_score=min(1.0, state.arrivals_last_min / max(1.0, k_cap)),
                    description=(
                        f"Arrival surge ({state.arrivals_last_min:.1f} cust/min) is passing through billing quickly ({b_cap:.1f} cap). "
                        f"It will hit the kitchen queue in ~{prop_lag} minutes, potentially exceeding kitchen capacity ({k_cap:.1f} ord/min)."
                    )
                ))
            else:
                prop_lag = max(2.0, state.billing_queue / b_cap)
                effects.append(PropagationEffect(
                    source_stage="Arrival",
                    target_stage="Billing Queue",
                    mechanism="Immediate Queue Accumulation",
                    time_lag_minutes=0.5,
                    predicted_impact_severity=DisruptionSeverity.HIGH,
                    vulnerability_score=0.9,
                    description=f"Arrival surge exceeds billing capacity ({state.arrivals_last_min:.1f} vs {b_cap:.1f}). Billing queue will build up rapidly."
                ))

        # 3. Kitchen Slowdown Propagation Analysis
        if DisruptionType.KITCHEN_SLOWDOWN in disruption_types or DisruptionType.CAPACITY_DEFICIT_KITCHEN in disruption_types:
            k_queue_drain_time = (state.kitchen_queue / k_cap) if k_cap > 0 else 15.0
            effects.append(PropagationEffect(
                source_stage="Kitchen",
                target_stage="Food Collection",
                mechanism="Backlog Stagnation & Collection Delay",
                time_lag_minutes=round(k_queue_drain_time, 1),
                predicted_impact_severity=DisruptionSeverity.CRITICAL if state.kitchen_queue > 12 else DisruptionSeverity.HIGH,
                vulnerability_score=min(1.0, state.kitchen_queue / 10.0),
                description=(
                    f"Kitchen preparation slowdown ({state.kitchen_service_time_sec:.0f}s per order) has caused a backlog of "
                    f"{state.kitchen_queue} orders. Orders face an estimated {k_queue_drain_time:.1f} min delay before collection."
                )
            ))

        return effects
