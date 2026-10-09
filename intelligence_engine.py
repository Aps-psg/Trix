"""
intelligence_engine.py - QueueFlux Central Intelligence Engine Orchestrator

Orchestrates all four intelligence sub-modules:
1. IntegrationAdapter: Normalizes heterogeneous teammate inputs to QueueFluxState.
2. DisruptionDetector: Identifies surges, slowdowns, and capacity deficits.
3. PropagationAnalyzer: Traces cross-stage ripple effects and time-lagged risks.
4. DecisionOptimizer: Recommends optimal resource-constrained interventions.

Produces dual outputs:
- Machine-readable dictionary/JSON for dashboard integration.
- Executive text summary for canteen managers.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from integration_adapter import QueueFluxState, IntegrationAdapter
from disruption_detector import DisruptionDetector, DisruptionAlert, DisruptionDetectorConfig
from propagation_analyzer import PropagationAnalyzer, PropagationEffect
from decision_optimizer import DecisionOptimizer, Intervention, ActionType


@dataclass
class DecisionReport:
    """Complete structured decision report produced by QueueFlux Intelligence Engine."""
    timestamp: str
    state: QueueFluxState
    adapter_warnings: List[str]
    detected_disruptions: List[DisruptionAlert]
    propagation_effects: List[PropagationEffect]
    recommended_action: Intervention
    alternative_actions: List[Intervention]
    baseline_wait_minutes: float
    expected_wait_minutes: float
    estimated_wait_reduction_minutes: float
    executive_summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable machine-readable output format."""
        return {
            "timestamp": self.timestamp,
            "state_summary": {
                "billing_queue": self.state.billing_queue,
                "kitchen_queue": self.state.kitchen_queue,
                "arrivals_last_min": self.state.arrivals_last_min,
                "billing_service_time_sec": self.state.billing_service_time_sec,
                "kitchen_service_time_sec": self.state.kitchen_service_time_sec,
                "active_billing_counters": self.state.active_billing_counters,
                "active_kitchen_servers": self.state.active_kitchen_servers,
                "available_extra_counters": self.state.available_extra_counters,
                "available_extra_workers": self.state.available_extra_workers,
                "uncertainty": self.state.uncertainty
            },
            "warnings": self.adapter_warnings,
            "detected_disruptions": [d.to_dict() for d in self.detected_disruptions],
            "propagation_effects": [p.to_dict() for p in self.propagation_effects],
            "recommended_action": self.recommended_action.to_dict(),
            "alternative_actions": [a.to_dict() for a in self.alternative_actions],
            "impact_comparison": {
                "baseline_wait_minutes": round(self.baseline_wait_minutes, 2),
                "expected_wait_minutes": round(self.expected_wait_minutes, 2),
                "estimated_wait_reduction_minutes": round(self.estimated_wait_reduction_minutes, 2),
            },
            "executive_summary": self.executive_summary
        }

    def to_human_summary(self) -> str:
        """Returns readable executive text summary for canteen management."""
        lines = []
        lines.append("================================================================================")
        lines.append(f"                 QUEUEFLUX INTELLIGENCE REPORT [{self.timestamp}]")
        lines.append("================================================================================")
        lines.append(f"CURRENT SYSTEM STATE:")
        lines.append(f"  * Billing Line:  {self.state.billing_queue} customers waiting | {self.state.active_billing_counters} open counters ({self.state.billing_service_time_sec:.0f}s/cust)")
        lines.append(f"  * Kitchen Queue: {self.state.kitchen_queue} orders waiting    | {self.state.active_kitchen_servers} active servers ({self.state.kitchen_service_time_sec:.0f}s/order)")
        lines.append(f"  * Arrival Rate:  {self.state.arrivals_last_min:.1f} cust/min (Normal: {self.state.normal_arrivals_per_min:.1f} cust/min)")
        lines.append(f"  * Resources:     {self.state.available_extra_counters} extra counters | {self.state.available_extra_workers} extra workers available")
        lines.append(f"  * Data Uncertainty Factor: {self.state.uncertainty:.2f}")

        if self.adapter_warnings:
            lines.append("\nINPUT DATA WARNINGS:")
            for w in self.adapter_warnings:
                lines.append(f"  ! {w}")

        lines.append("\n1. DETECTED DISRUPTIONS:")
        if self.detected_disruptions:
            for d in self.detected_disruptions:
                lines.append(f"  [{d.severity}] {d.disruption_type} at {d.stage} Stage (Conf: {d.confidence:.0%})")
                lines.append(f"     -> {d.description}")
        else:
            lines.append("  [OK] No operational disruptions detected. Operations running normally.")

        lines.append("\n2. CROSS-STAGE PROPAGATION RISKS:")
        if self.propagation_effects:
            for p in self.propagation_effects:
                lines.append(f"  -> [{p.predicted_impact_severity}] {p.source_stage} -> {p.target_stage} (Lag: ~{p.time_lag_minutes:.1f} min)")
                lines.append(f"     {p.description}")
        else:
            lines.append("  [OK] No cross-stage propagation threats identified.")

        lines.append("\n3. RECOMMENDED ACTION & IMPACT:")
        rec = self.recommended_action
        lines.append(f"  RECOMMENDED: {rec.action_name} (Target: {rec.target_stage})")
        lines.append(f"  Reasoning:   {rec.reasoning}")
        lines.append(f"  Constraints: {rec.constraint_notes}")
        lines.append(f"  Baseline Wait Time (No Action): {self.baseline_wait_minutes:.1f} minutes")
        lines.append(f"  Projected Wait Time (With Action): {self.expected_wait_minutes:.1f} minutes")
        lines.append(f"  EXPECTED SAVINGS: {self.estimated_wait_reduction_minutes:.1f} minutes (Estimated)")

        lines.append("\n4. ALTERNATIVE ACTIONS EVALUATED:")
        for alt in self.alternative_actions:
            if alt.action_name != rec.action_name:
                feas = "FEASIBLE" if alt.is_feasible else "INFEASIBLE"
                lines.append(f"  * {alt.action_name:<30} | {feas:<10} | Wait: {alt.expected_wait_time_min:.1f}m | Savings: {alt.wait_reduction_min:.1f}m")

        lines.append("================================================================================")
        return "\n".join(lines)


class IntelligenceEngine:
    """
    Central Coordinator for QueueFlux Intelligence Pipeline.
    """

    def __init__(self, detector_config: Optional[DisruptionDetectorConfig] = None):
        self.detector = DisruptionDetector(detector_config)
        self.propagation_analyzer = PropagationAnalyzer()
        self.optimizer = DecisionOptimizer()

    def process_state(self, state: QueueFluxState, adapter_warnings: Optional[List[str]] = None) -> DecisionReport:
        """
        Processes a clean QueueFluxState object through the intelligence pipeline.
        """
        warnings = adapter_warnings if adapter_warnings is not None else []

        # Step 1: Detect Disruptions
        disruptions = self.detector.detect_disruptions(state)

        # Step 2: Analyze Cross-Stage Propagation
        propagation = self.propagation_analyzer.analyze_propagation(state, disruptions)

        # Step 3: Optimize Decision / Interventions
        best_action, all_candidates = self.optimizer.optimize_decision(state, disruptions, propagation)

        # Step 4: Baseline comparison calculation
        baseline_wait = state.calculate_estimated_wait_time_min()
        expected_wait = best_action.expected_wait_time_min
        reduction = max(0.0, baseline_wait - expected_wait)

        # Construct Executive Summary string
        if best_action.action_name == ActionType.NO_ACTION:
            exec_summary = f"Operations normal or constrained. Maintain baseline state. Expected wait time: {baseline_wait:.1f} min."
        else:
            exec_summary = (
                f"Action recommended: {best_action.action_name}. Resolves bottleneck in {best_action.target_stage}. "
                f"Reduces customer wait time from {baseline_wait:.1f} min to {expected_wait:.1f} min (saving {reduction:.1f} min)."
            )

        return DecisionReport(
            timestamp=state.timestamp,
            state=state,
            adapter_warnings=warnings,
            detected_disruptions=disruptions,
            propagation_effects=propagation,
            recommended_action=best_action,
            alternative_actions=all_candidates,
            baseline_wait_minutes=baseline_wait,
            expected_wait_minutes=expected_wait,
            estimated_wait_reduction_minutes=reduction,
            executive_summary=exec_summary
        )

    def process_raw_input(
        self,
        person1_data: Optional[Dict[str, Any]] = None,
        person2_data: Optional[Dict[str, Any]] = None,
        custom_dict: Optional[Dict[str, Any]] = None
    ) -> DecisionReport:
        """
        Accepts heterogeneous dictionary inputs from Person 1 & Person 2,
        adapts them using IntegrationAdapter, and runs the intelligence pipeline.
        """
        state, warnings = IntegrationAdapter.adapt_raw_input(
            person1_data=person1_data,
            person2_data=person2_data,
            custom_dict=custom_dict
        )
        return self.process_state(state, adapter_warnings=warnings)
