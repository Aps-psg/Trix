"""
Uncertainty-Aware Intervention Optimization Engine for QueueFlux.
Formulates risk-sensitive stochastic optimization over operational interventions.
Evaluates actions across particle distributions and forward rollouts using
Expected Total Delay, Conditional Value-at-Risk (CVaR_0.95), and Reneging Loss.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from queueflux.config import SystemConfig, DEFAULT_CONFIG
from queueflux.models.state import EstimatedState
from queueflux.propagation.cross_stage import PropagationRiskReport


class InterventionAction(str, Enum):
    DO_NOTHING = "DO_NOTHING"
    OPEN_BILLING_COUNTER = "OPEN_BILLING_COUNTER"
    REALLOCATE_STAFF_TO_KITCHEN = "REALLOCATE_STAFF_TO_KITCHEN"
    ACTIVATE_KITCHEN_BATCHING = "ACTIVATE_KITCHEN_BATCHING"
    BROADCAST_WAIT_WARNING = "BROADCAST_WAIT_WARNING"


@dataclass
class ActionEvaluation:
    """Evaluated metric outcome for candidate intervention action."""
    action: InterventionAction
    expected_delay_minutes: float
    cvar_95_delay: float
    expected_reneged_customers: float
    operational_cost: float
    total_risk_adjusted_loss: float
    expected_wait_reduction_pct: float
    rationale: str


class InterventionOptimizer:
    """
    Evaluates candidate operational actions under posterior state uncertainty.
    Selects the policy action minimizing total risk-adjusted societal cost.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def optimize_action(self, state: EstimatedState, 
                        propagation_report: Optional[PropagationRiskReport] = None,
                        cooldown_active: bool = False) -> Tuple[InterventionAction, ActionEvaluation, List[ActionEvaluation]]:
        """
        Evaluates all candidate actions under state uncertainty.
        Returns:
            best_action: Chosen action enum
            best_eval: Detailed evaluation of the winning action
            all_evals: Comparison table of all candidate evaluations
        """
        candidate_actions = [
            InterventionAction.DO_NOTHING,
            InterventionAction.OPEN_BILLING_COUNTER,
            InterventionAction.REALLOCATE_STAFF_TO_KITCHEN,
            InterventionAction.ACTIVATE_KITCHEN_BATCHING,
            InterventionAction.BROADCAST_WAIT_WARNING
        ]
        
        # Fast-path: If system is operating comfortably under nominal capacity, DO_NOTHING
        if (state.Q_B_hat < 6.0 and state.Q_K_hat < 8.0 
                and state.most_likely_regime.value == "NORMAL"
                and (propagation_report is None or propagation_report.downstream_surge_shockwave_risk < 0.4)):
            base_eval = ActionEvaluation(
                action=InterventionAction.DO_NOTHING,
                expected_delay_minutes=round(float(state.W_B_hat + state.W_K_hat), 1),
                cvar_95_delay=round(float((state.W_B_hat + state.W_K_hat) * 1.2), 1),
                expected_reneged_customers=0.0,
                operational_cost=0.0,
                total_risk_adjusted_loss=round(float(state.W_B_hat + state.W_K_hat), 1),
                expected_wait_reduction_pct=0.0,
                rationale="System operating within nominal capacity; queues clear naturally."
            )
            return InterventionAction.DO_NOTHING, base_eval, [base_eval]

        evaluations: List[ActionEvaluation] = []
        
        # Base baseline performance
        base_eval = self._evaluate_single_action(InterventionAction.DO_NOTHING, state, propagation_report)
        evaluations.append(base_eval)
        
        for act in candidate_actions:
            if act == InterventionAction.DO_NOTHING:
                continue
            ev = self._evaluate_single_action(act, state, propagation_report, baseline_loss=base_eval.expected_delay_minutes)
            evaluations.append(ev)
            
        # If cooldown is active, penalize switching actions
        if cooldown_active:
            for ev in evaluations:
                if ev.action != InterventionAction.DO_NOTHING:
                    ev.total_risk_adjusted_loss += 50.0  # inertia penalty
                    
        # Select action with minimum total risk-adjusted loss
        best_eval = min(evaluations, key=lambda e: e.total_risk_adjusted_loss)
        return best_eval.action, best_eval, evaluations

    def _evaluate_single_action(self, action: InterventionAction, state: EstimatedState, 
                                propagation_report: Optional[PropagationRiskReport],
                                baseline_loss: float = 0.0) -> ActionEvaluation:
        """
        Simulates outcome distribution of the action over a 15-minute lookahead window.
        Uses Monte Carlo sampling over state credible interval.
        """
        n_samples = 80
        
        # Draw samples of initial queues from belief percentiles
        sig_b = max(0.5, (state.Q_B_p90 - state.Q_B_p10) / 2.56)
        sig_k = max(0.5, (state.Q_K_p90 - state.Q_K_p10) / 2.56)
        qb_s = np.maximum(0.0, self.rng.normal(state.Q_B_hat, sig_b, n_samples))
        qk_s = np.maximum(0.0, self.rng.normal(state.Q_K_hat, sig_k, n_samples))
        
        # Effective resources under action
        c_b = state.C_B_hat
        c_k = state.C_K_hat
        op_cost = 0.0
        eff_mult_k = 1.0
        arr_mult = 1.0
        
        if action == InterventionAction.DO_NOTHING:
            op_cost = 0.0
            rationale = "System operating within normal bounds; no intervention overhead needed."
            
        elif action == InterventionAction.OPEN_BILLING_COUNTER:
            c_b += 1.0
            op_cost = self.config.cost_counter_activation
            rationale = "High billing queue detected; activating additional billing counter to clear upstream bottleneck."
            
        elif action == InterventionAction.REALLOCATE_STAFF_TO_KITCHEN:
            if c_b > 1.0:
                c_b -= 1.0
                c_k += 1.0
                op_cost = self.config.cost_staff_shift_penalty
                rationale = "Severe downstream kitchen backlog; shifting cross-trained cashier to kitchen assembly."
            else:
                op_cost = 1e4
                rationale = "Cannot reallocate: minimum billing capacity reached."
                
        elif action == InterventionAction.ACTIVATE_KITCHEN_BATCHING:
            eff_mult_k = 1.25  # 25% throughput boost through batching
            op_cost = self.config.cost_express_batch_penalty
            rationale = "Activating high-throughput batching for standard meals to relieve kitchen congestion."
            
        elif action == InterventionAction.BROADCAST_WAIT_WARNING:
            arr_mult = 0.80   # 20% incoming arrival dampening
            op_cost = 5.0
            rationale = "Notifying campus app of high canteen wait times to dampen incoming arrival surge."

        # Simulate 10-minute queue progression
        regime = state.most_likely_regime.value
        reg_eff_b = self.config.regime_service_multiplier_billing.get(regime, 1.0)
        reg_eff_k = self.config.regime_service_multiplier_kitchen.get(regime, 1.0)
        
        mu_b = (c_b / 0.95) * reg_eff_b
        mu_k = (c_k / 3.60) * reg_eff_k * eff_mult_k
        
        total_customer_delays = np.zeros(n_samples)
        total_reneges = np.zeros(n_samples)
        
        for step in range(10):
            # Arrival step
            lambda_step = 3.0 * arr_mult
            arrs = self.rng.poisson(lambda_step, n_samples)
            
            # Billing service
            serv_b = np.minimum(qb_s + arrs, self.rng.poisson(max(0.1, mu_b), n_samples))
            qb_s = np.maximum(0.0, qb_s + arrs - serv_b)
            
            # Kitchen service
            serv_k = np.minimum(qk_s + serv_b, self.rng.poisson(max(0.1, mu_k), n_samples))
            qk_s = np.maximum(0.0, qk_s + serv_b - serv_k)
            
            # Estimate delay accumulated in step
            total_customer_delays += (qb_s * 0.95 + qk_s * 3.60)
            
            # Reneging risk
            renege_prob_b = np.clip((qb_s - 15.0) / 30.0, 0.0, 0.5)
            renege_prob_k = np.clip((qk_s - 20.0) / 40.0, 0.0, 0.5)
            total_reneges += (renege_prob_b * qb_s + renege_prob_k * qk_s) * 0.1
            
        mean_delay = float(np.mean(total_customer_delays))
        # CVaR at 95%
        q95 = np.percentile(total_customer_delays, 95)
        cvar_95 = float(np.mean(total_customer_delays[total_customer_delays >= q95]))
        mean_reneges = float(np.mean(total_reneges))
        
        # Risk-adjusted loss function:
        # Loss = E[Delay] + lambda * CVaR + penalty_renege * Reneges + OpCost
        total_loss = (
            mean_delay * self.config.cost_wait_minute_customer
            + self.config.risk_aversion_lambda * (cvar_95 - mean_delay)
            + mean_reneges * self.config.cost_cancellation_penalty
            + op_cost
        )
        
        wait_reduction_pct = 0.0
        if baseline_loss > 0:
            wait_reduction_pct = max(0.0, (baseline_loss - mean_delay) / baseline_loss * 100.0)
            
        return ActionEvaluation(
            action=action,
            expected_delay_minutes=round(mean_delay, 1),
            cvar_95_delay=round(cvar_95, 1),
            expected_reneged_customers=round(mean_reneges, 2),
            operational_cost=round(op_cost, 1),
            total_risk_adjusted_loss=round(total_loss, 1),
            expected_wait_reduction_pct=round(wait_reduction_pct, 1),
            rationale=rationale
        )
