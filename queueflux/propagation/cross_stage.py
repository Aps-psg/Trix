"""
Cross-Stage Disruption Propagation Engine for QueueFlux.
Models the dynamical cross-coupling between Stage A (Billing) and Stage B (Kitchen Preparation).
Quantifies:
1. Starvation-to-Surge shockwave transmission (Stage A slowdown -> Stage B starve -> Stage B burst)
2. Backlog pressure & reneging feedback (Stage B bottleneck -> cancelation spike -> hall congestion)
3. Dynamic Propagation Sensitivity Matrix Phi_{A -> B}(tau)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from queueflux.config import SystemConfig, DEFAULT_CONFIG
from queueflux.models.state import EstimatedState


@dataclass
class PropagationRiskReport:
    """Diagnostic report on cross-stage vulnerability."""
    downstream_starvation_risk: float       # [0, 1] Risk of kitchen sitting idle due to billing choke
    downstream_surge_shockwave_risk: float  # [0, 1] Risk of delayed tsunami of orders hitting kitchen
    upstream_backpressure_risk: float       # [0, 1] Risk of kitchen bottleneck triggering billing cancellations
    cross_stage_sensitivity_peak_lag: int   # Minutes until upstream shock reaches maximum downstream impact
    propagation_sensitivity_peak: float     # d(W_K)/d(W_B) sensitivity coefficient
    recommended_stage_focus: str            # "STAGE_A_BILLING", "STAGE_B_KITCHEN", or "BALANCED"


class CrossStagePropagationEngine:
    """
    Analyzes and models cross-stage transmission of operational shocks.
    Enables QueueFlux to intervene prophylactically before an upstream delay
    cascades into a massive downstream kitchen backlog.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.config = config
        self.max_lag_tau = 15  # Up to 15 minutes lag horizon

    def compute_impulse_response_matrix(self, state: EstimatedState) -> np.ndarray:
        """
        Computes the discrete impulse response kernel Phi_{A -> B}(tau)
        representing the fraction of an incremental upstream billing workload delta(W_B)
        that will arrive as active kitchen backlog at lag tau minutes.
        """
        tau_range = np.arange(1, self.max_lag_tau + 1)
        
        # Mean clearance time of billing queue
        c_b = max(1.0, state.C_B_hat)
        eff_b = self.config.regime_service_multiplier_billing.get(state.most_likely_regime.value, 1.0)
        mu_b = c_b * eff_b
        
        # Billing queue drainage follows Gamma / Erlang clearing curve
        shape_k = max(1.5, state.Q_B_hat / 2.0)
        scale_theta = max(0.5, (state.W_B_hat / mu_b) / shape_k) if shape_k > 0 else 1.0
        
        # Gamma PDF evaluated over lag tau
        from scipy import stats
        phi_values = stats.gamma.pdf(tau_range, a=shape_k, scale=scale_theta)
        
        # Scale by conversion to kitchen workload (1 billing transaction generates ~3.6 min of prep workload)
        workload_ratio = 3.60 / 0.95
        phi_scaled = phi_values * workload_ratio
        
        return phi_scaled

    def evaluate_propagation_risk(self, state: EstimatedState) -> PropagationRiskReport:
        """
        Evaluates current system state for cascading disruption risks.
        """
        phi_curve = self.compute_impulse_response_matrix(state)
        peak_lag = int(np.argmax(phi_curve) + 1)
        peak_sensitivity = float(np.max(phi_curve))
        
        c_b = max(1.0, state.C_B_hat)
        c_k = max(1.0, state.C_K_hat)
        
        # 1. Downstream Starvation Risk:
        # Occurs if billing is heavily slowed down while kitchen has low current queue
        eff_b = self.config.regime_service_multiplier_billing.get(state.most_likely_regime.value, 1.0)
        starvation_score = 0.0
        if eff_b < 0.6 and state.Q_K_hat < 3.0:
            starvation_score = min(1.0, (0.6 - eff_b) / 0.4 + (3.0 - state.Q_K_hat) / 3.0)
            
        # 2. Downstream Surge Shockwave Risk:
        # High billing queue W_B waiting to be cleared, which will soon overwhelm kitchen capacity
        kitchen_drain_rate = c_k * self.config.regime_service_multiplier_kitchen.get(state.most_likely_regime.value, 1.0)
        pending_incoming_kitchen_work = state.W_B_hat * (3.60 / 0.95)
        surge_ratio = pending_incoming_kitchen_work / (kitchen_drain_rate * 5.0)  # over next 5 mins
        surge_risk = float(np.clip((surge_ratio - 0.8) / 1.5, 0.0, 1.0))
        
        # 3. Upstream Backpressure Risk:
        # High kitchen queue W_K leading to collection shelf congestion and customer reneging
        backpressure_risk = float(np.clip((state.W_K_hat - 15.0) / 25.0, 0.0, 1.0))
        
        # Recommend intervention focus
        if surge_risk > 0.6 or backpressure_risk > 0.5:
            focus = "STAGE_B_KITCHEN"
        elif starvation_score > 0.5 or (state.W_B_hat > 12.0 and state.W_K_hat < 15.0):
            focus = "STAGE_A_BILLING"
        else:
            focus = "BALANCED"
            
        return PropagationRiskReport(
            downstream_starvation_risk=round(starvation_score, 3),
            downstream_surge_shockwave_risk=round(surge_risk, 3),
            upstream_backpressure_risk=round(backpressure_risk, 3),
            cross_stage_sensitivity_peak_lag=peak_lag,
            propagation_sensitivity_peak=round(peak_sensitivity, 3),
            recommended_stage_focus=focus
        )
