"""
decision_optimizer.py - QueueFlux Decision Optimizer Module

Recommends practical, resource-constrained interventions to canteen management
to minimize total customer waiting time.

Feasible Interventions:
1. NO_ACTION: Maintain current operational state.
2. OPEN_EXTRA_BILLING_COUNTER: Open an additional billing counter (costs 1 counter + 1 worker).
3. ALLOCATE_WORKER_TO_KITCHEN: Add a floater worker to kitchen prep (costs 1 worker).
4. REALLOCATE_COUNTER_TO_KITCHEN: Shift 1 staff from billing to kitchen (costs 0 extra workers).
5. PREPARE_RUSH_STAGING: Pre-batch high-demand food items for anticipated surge.

Enforces strict resource constraints and checks for secondary bottleneck risks.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from integration_adapter import QueueFluxState
from disruption_detector import DisruptionAlert, DisruptionType
from propagation_analyzer import PropagationEffect


class ActionType:
    NO_ACTION = "NO_ACTION"
    OPEN_EXTRA_BILLING_COUNTER = "OPEN_EXTRA_BILLING_COUNTER"
    ALLOCATE_WORKER_TO_KITCHEN = "ALLOCATE_WORKER_TO_KITCHEN"
    REALLOCATE_COUNTER_TO_KITCHEN = "REALLOCATE_COUNTER_TO_KITCHEN"
    PREPARE_RUSH_STAGING = "PREPARE_RUSH_STAGING"


@dataclass
class Intervention:
    """Represents a evaluated candidate action for canteen operations."""
    action_name: str
    target_stage: str
    resource_cost_counters: int
    resource_cost_workers: int
    expected_wait_time_min: float
    wait_reduction_min: float
    confidence: float
    is_feasible: bool
    constraint_notes: str
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "target_stage": self.target_stage,
            "resource_cost_counters": self.resource_cost_counters,
            "resource_cost_workers": self.resource_cost_workers,
            "expected_wait_time_min": round(self.expected_wait_time_min, 2),
            "wait_reduction_min": round(self.wait_reduction_min, 2),
            "confidence": round(self.confidence, 2),
            "is_feasible": self.is_feasible,
            "constraint_notes": self.constraint_notes,
            "reasoning": self.reasoning,
        }


class DecisionOptimizer:
    """
    Evaluates feasible candidate interventions against queuing theory models,
    resource limits, and secondary bottleneck constraints.
    """

    def optimize_decision(
        self,
        state: QueueFluxState,
        active_disruptions: List[DisruptionAlert],
        propagation_effects: List[PropagationEffect]
    ) -> Tuple[Intervention, List[Intervention]]:
        """
        Evaluates all candidate actions and returns (best_recommended_action, all_evaluated_candidates).
        """
        candidates: List[Intervention] = []

        # 1. Baseline: NO_ACTION
        baseline_action = self._evaluate_no_action(state)
        candidates.append(baseline_action)

        # 2. Action: OPEN_EXTRA_BILLING_COUNTER
        open_counter_action = self._evaluate_open_extra_counter(state)
        candidates.append(open_counter_action)

        # 3. Action: ALLOCATE_WORKER_TO_KITCHEN
        add_kitchen_worker_action = self._evaluate_allocate_kitchen_worker(state)
        candidates.append(add_kitchen_worker_action)

        # 4. Action: REALLOCATE_COUNTER_TO_KITCHEN
        reallocate_action = self._evaluate_reallocate_counter_to_kitchen(state)
        candidates.append(reallocate_action)

        # 5. Action: PREPARE_RUSH_STAGING
        rush_staging_action = self._evaluate_rush_staging(state, active_disruptions)
        candidates.append(rush_staging_action)

        # Filter feasible candidates
        feasible_candidates = [c for c in candidates if c.is_feasible]

        # Uncertainty Penalty adjustment
        if state.uncertainty > 0.3:
            for c in feasible_candidates:
                c.confidence = max(0.1, c.confidence * (1.0 - state.uncertainty * 0.5))

        # Select best action based on maximum wait time reduction
        # (Must exceed NO_ACTION by at least 0.35 mins to justify operational disruption)
        best_action = baseline_action
        max_reduction = 0.35

        for c in feasible_candidates:
            if c.action_name != ActionType.NO_ACTION and c.wait_reduction_min > max_reduction:
                max_reduction = c.wait_reduction_min
                best_action = c

        return best_action, candidates

    def _evaluate_no_action(self, state: QueueFluxState) -> Intervention:
        est_wait = state.calculate_estimated_wait_time_min()
        base_conf = max(0.1, min(1.0, 1.0 - state.uncertainty))
        return Intervention(
            action_name=ActionType.NO_ACTION,
            target_stage="System",
            resource_cost_counters=0,
            resource_cost_workers=0,
            expected_wait_time_min=est_wait,
            wait_reduction_min=0.0,
            confidence=base_conf,
            is_feasible=True,
            constraint_notes="No resource constraints required.",
            reasoning=f"Maintain current state. Baseline estimated waiting time is {est_wait:.1f} minutes."
        )

    def _evaluate_open_extra_counter(self, state: QueueFluxState) -> Intervention:
        has_counter = state.available_extra_counters >= 1
        has_worker = state.available_extra_workers >= 1
        is_feasible = has_counter and has_worker

        notes = []
        if not has_counter:
            notes.append(f"Insufficient extra counters (available: {state.available_extra_counters}, required: 1)")
        if not has_worker:
            notes.append(f"Insufficient extra staff (available: {state.available_extra_workers}, required: 1)")

        constraint_str = "; ".join(notes) if notes else "All required resources available (1 counter, 1 worker)."

        # Compute new wait time with +1 billing counter
        temp_state = QueueFluxState(
            billing_queue=state.billing_queue,
            kitchen_queue=state.kitchen_queue,
            billing_service_time_sec=state.billing_service_time_sec,
            kitchen_service_time_sec=state.kitchen_service_time_sec,
            active_billing_counters=state.active_billing_counters + 1,
            active_kitchen_servers=state.active_kitchen_servers
        )
        new_wait = temp_state.calculate_estimated_wait_time_min()

        # Secondary bottleneck check: If kitchen queue is already heavily congested, opening billing counter makes kitchen worse!
        k_cap = state.get_kitchen_capacity_per_min()
        b_cap_new = temp_state.get_billing_capacity_per_min()
        secondary_bottleneck = False

        if b_cap_new > k_cap and state.kitchen_queue > 8:
            secondary_bottleneck = True
            constraint_str += " WARNING: Risk of secondary bottleneck in Kitchen."
            # Penalize effective wait time reduction because orders will stall in kitchen
            new_wait += (state.kitchen_queue * 0.15)

        base_wait = state.calculate_estimated_wait_time_min()
        reduction = max(0.0, base_wait - new_wait)
        conf = max(0.1, 0.9 - state.uncertainty)

        if secondary_bottleneck:
            conf *= 0.7

        reasoning = (
            f"Open 1 additional billing counter (total: {state.active_billing_counters + 1}). "
            f"Expected wait time drops from {base_wait:.1f}m to {new_wait:.1f}m (saving {reduction:.1f}m)."
        )

        return Intervention(
            action_name=ActionType.OPEN_EXTRA_BILLING_COUNTER,
            target_stage="Billing",
            resource_cost_counters=1,
            resource_cost_workers=1,
            expected_wait_time_min=new_wait,
            wait_reduction_min=reduction,
            confidence=conf,
            is_feasible=is_feasible,
            constraint_notes=constraint_str,
            reasoning=reasoning
        )

    def _evaluate_allocate_kitchen_worker(self, state: QueueFluxState) -> Intervention:
        has_worker = state.available_extra_workers >= 1
        constraint_str = "Required 1 extra worker available." if has_worker else f"Insufficient extra staff (available: {state.available_extra_workers}, required: 1)."

        # Compute new wait time with +1 kitchen server
        temp_state = QueueFluxState(
            billing_queue=state.billing_queue,
            kitchen_queue=state.kitchen_queue,
            billing_service_time_sec=state.billing_service_time_sec,
            kitchen_service_time_sec=state.kitchen_service_time_sec,
            active_billing_counters=state.active_billing_counters,
            active_kitchen_servers=state.active_kitchen_servers + 1
        )
        new_wait = temp_state.calculate_estimated_wait_time_min()

        base_wait = state.calculate_estimated_wait_time_min()
        reduction = max(0.0, base_wait - new_wait)
        conf = max(0.1, 0.9 - state.uncertainty)

        reasoning = (
            f"Deploy 1 extra floater worker to kitchen (total servers: {state.active_kitchen_servers + 1}). "
            f"Expected wait time drops from {base_wait:.1f}m to {new_wait:.1f}m (saving {reduction:.1f}m)."
        )

        return Intervention(
            action_name=ActionType.ALLOCATE_WORKER_TO_KITCHEN,
            target_stage="Kitchen",
            resource_cost_counters=0,
            resource_cost_workers=1,
            expected_wait_time_min=new_wait,
            wait_reduction_min=reduction,
            confidence=conf,
            is_feasible=has_worker,
            constraint_notes=constraint_str,
            reasoning=reasoning
        )

    def _evaluate_reallocate_counter_to_kitchen(self, state: QueueFluxState) -> Intervention:
        # Requires at least 2 billing counters active so billing isn't shut down
        can_reallocate = (state.active_billing_counters >= 2)
        b_cap = state.get_billing_capacity_per_min()

        # Only feasible if billing line is short/empty AND kitchen is bottlenecked
        billing_has_surplus = (state.billing_queue <= 3) and (b_cap >= state.arrivals_last_min * 1.2)
        is_feasible = can_reallocate and billing_has_surplus

        constraint_notes = "Billing has surplus capacity; counter staff can be shifted to kitchen." if is_feasible else (
            f"Infeasible: Active billing counters = {state.active_billing_counters} (min 2 required) "
            f"or billing queue too long ({state.billing_queue} cust)."
        )

        temp_state = QueueFluxState(
            billing_queue=state.billing_queue,
            kitchen_queue=state.kitchen_queue,
            billing_service_time_sec=state.billing_service_time_sec,
            kitchen_service_time_sec=state.kitchen_service_time_sec,
            active_billing_counters=max(1, state.active_billing_counters - 1),
            active_kitchen_servers=state.active_kitchen_servers + 1
        )
        new_wait = temp_state.calculate_estimated_wait_time_min()
        base_wait = state.calculate_estimated_wait_time_min()
        reduction = max(0.0, base_wait - new_wait)
        conf = max(0.1, 0.85 - state.uncertainty)

        reasoning = (
            f"Reallocate 1 cashier from billing to kitchen prep. Reduces billing counters to {temp_state.active_billing_counters} "
            f"and increases kitchen servers to {temp_state.active_kitchen_servers}. "
            f"Expected wait time drops from {base_wait:.1f}m to {new_wait:.1f}m (saving {reduction:.1f}m)."
        )

        return Intervention(
            action_name=ActionType.REALLOCATE_COUNTER_TO_KITCHEN,
            target_stage="Kitchen",
            resource_cost_counters=0,
            resource_cost_workers=0,  # Reallocates existing staff
            expected_wait_time_min=new_wait,
            wait_reduction_min=reduction,
            confidence=conf,
            is_feasible=is_feasible,
            constraint_notes=constraint_notes,
            reasoning=reasoning
        )

    def _evaluate_rush_staging(self, state: QueueFluxState, active_disruptions: List[DisruptionAlert]) -> Intervention:
        has_surge = any(d.disruption_type == DisruptionType.ARRIVAL_SURGE for d in active_disruptions)
        has_forecast_surge = max(state.forecast_billing_queue + state.forecast_kitchen_queue + [0]) > 10

        is_feasible = has_surge or has_forecast_surge

        # Pre-staging reduces kitchen service time by 20%
        staged_kitchen_sec = max(30.0, state.kitchen_service_time_sec * 0.80)
        temp_state = QueueFluxState(
            billing_queue=state.billing_queue,
            kitchen_queue=state.kitchen_queue,
            billing_service_time_sec=state.billing_service_time_sec,
            kitchen_service_time_sec=staged_kitchen_sec,
            active_billing_counters=state.active_billing_counters,
            active_kitchen_servers=state.active_kitchen_servers
        )
        new_wait = temp_state.calculate_estimated_wait_time_min()
        base_wait = state.calculate_estimated_wait_time_min()
        reduction = max(0.0, base_wait - new_wait)
        conf = max(0.1, 0.80 - state.uncertainty)

        constraint_notes = "Pre-batching rush items requires kitchen prep lead time." if is_feasible else "No surge detected or forecasted; rush staging not required."

        reasoning = (
            f"Pre-stage popular items ahead of rush hour. Reduces kitchen prep time to {staged_kitchen_sec:.0f}s per order. "
            f"Expected wait time drops from {base_wait:.1f}m to {new_wait:.1f}m (saving {reduction:.1f}m)."
        )

        return Intervention(
            action_name=ActionType.PREPARE_RUSH_STAGING,
            target_stage="Kitchen",
            resource_cost_counters=0,
            resource_cost_workers=0,
            expected_wait_time_min=new_wait,
            wait_reduction_min=reduction,
            confidence=conf,
            is_feasible=is_feasible,
            constraint_notes=constraint_notes,
            reasoning=reasoning
        )
