"""
Workload-Aware Waiting-Time Predictor for QueueFlux.
Computes waiting time predictions by explicitly separating queue entity count,
heterogeneous order workloads, active server capacities, residual service times,
and dynamic regime degradations.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from queueflux.config import SystemConfig, DEFAULT_CONFIG
from queueflux.models.entities import OrderType, RegimeType
from queueflux.models.state import EstimatedState


@dataclass
class WaitingTimePrediction:
    """Detailed prediction breakdown for a customer/order."""
    expected_billing_wait: float          # Time waiting in billing queue
    billing_wait_p10: float
    billing_wait_p90: float
    
    expected_kitchen_wait: float          # Time waiting in kitchen queue
    kitchen_wait_p10: float
    kitchen_wait_p90: float
    
    expected_total_system_time: float     # Wait + Billing service + Kitchen service
    total_system_time_p10: float
    total_system_time_p90: float
    
    regime_adjustment_factor: float       # Multiplier applied due to detected regime


class WorkloadWaitingTimePredictor:
    """
    Workload-Aware Waiting Time Prediction Model.
    Unlike naive FIFO queue length models (which assume wait = N * avg_service),
    QueueFlux accounts for:
    1. In-service residual service load R_t
    2. Sum of heterogeneous order workloads W_t
    3. Concurrency / server counts C_t
    4. Regime-dependent service efficiency mu(Z_t)
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.config = config

    def predict(self, state: EstimatedState, order_type: OrderType) -> WaitingTimePrediction:
        """
        Generates calibrated waiting time predictions using EstimatedState.
        Zero access to ground truth.
        """
        # Regime efficiency multipliers
        regime = state.most_likely_regime.value
        eff_b = self.config.regime_service_multiplier_billing.get(regime, 1.0)
        eff_k = self.config.regime_service_multiplier_kitchen.get(regime, 1.0)
        
        c_b = max(1.0, state.C_B_hat)
        c_k = max(1.0, state.C_K_hat)
        
        # Effective capacity in workload-minutes per wall-clock minute
        rate_b = c_b * eff_b
        rate_k = c_k * eff_k
        
        # Billing wait: (W_B + residual_B) / rate_b
        expected_wait_b = (state.W_B_hat + state.residual_B_hat) / rate_b
        p10_wait_b = max(0.0, (state.W_B_p10 + state.residual_B_hat * 0.5) / rate_b)
        p90_wait_b = max(0.0, (state.W_B_p90 + state.residual_B_hat * 1.5) / rate_b)
        
        # Kitchen wait: (W_K + residual_K) / rate_k
        expected_wait_k = (state.W_K_hat + state.residual_K_hat) / rate_k
        p10_wait_k = max(0.0, (state.W_K_p10 + state.residual_K_hat * 0.5) / rate_k)
        p90_wait_k = max(0.0, (state.W_K_p90 + state.residual_K_hat * 1.5) / rate_k)
        
        # Customer's own service durations
        params = self.config.service_params[order_type.value]
        own_b_service = params["billing"][0] / eff_b
        own_k_service = params["kitchen"][0] / eff_k
        
        # Total time from entry to food collection
        total_expected = expected_wait_b + own_b_service + expected_wait_k + own_k_service
        total_p10 = p10_wait_b + (own_b_service * 0.8) + p10_wait_k + (own_k_service * 0.8)
        total_p90 = p90_wait_b + (own_b_service * 1.3) + p90_wait_k + (own_k_service * 1.3)
        
        return WaitingTimePrediction(
            expected_billing_wait=round(float(expected_wait_b), 2),
            billing_wait_p10=round(float(p10_wait_b), 2),
            billing_wait_p90=round(float(p90_wait_b), 2),
            expected_kitchen_wait=round(float(expected_wait_k), 2),
            kitchen_wait_p10=round(float(p10_wait_k), 2),
            kitchen_wait_p90=round(float(p90_wait_k), 2),
            expected_total_system_time=round(float(total_expected), 2),
            total_system_time_p10=round(float(total_p10), 2),
            total_system_time_p90=round(float(total_p90), 2),
            regime_adjustment_factor=round(float(1.0 / min(eff_b, eff_k)), 2)
        )
