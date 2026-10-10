"""
Five-minute probabilistic queue forecaster for QueueFlux.
Performs multi-step stochastic rollouts across horizons k in {1, 2, 3, 4, 5} minutes.
Outputs median, P10, and P90 trajectory quantiles, with CRPS metric evaluation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from queueflux.config import SystemConfig, DEFAULT_CONFIG
from queueflux.models.state import EstimatedState


@dataclass
class HorizonForecast:
    """Probabilistic forecast for a specific future horizon minute (e.g. t+1 to t+5)."""
    horizon_minutes: int
    
    Q_B_p10: float
    Q_B_median: float
    Q_B_p90: float
    
    Q_K_p10: float
    Q_K_median: float
    Q_K_p90: float
    
    W_B_median: float
    W_K_median: float


class FiveMinuteQueueForecaster:
    """
    Five-minute rolling probabilistic forecaster.
    Executes vectorized stochastic rollouts over the current particle state belief.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG, num_rollout_samples: int = 300, seed: int = 42):
        self.config = config
        self.num_rollout_samples = num_rollout_samples
        self.rng = np.random.default_rng(seed)
        self.horizons = [1, 2, 3, 4, 5]

    def forecast_5min(self, state: EstimatedState, current_minute_of_day: float) -> List[HorizonForecast]:
        """
        Projects queue states forward from t to t+5 minutes.
        Uses belief quantiles as starting distribution and projects transitions.
        """
        # Sample starting particles from estimated state belief
        # Truncated normal centered around hat with width (p90 - p10)/2.56
        sigma_qb = max(0.5, (state.Q_B_p90 - state.Q_B_p10) / 2.56)
        sigma_qk = max(0.5, (state.Q_K_p90 - state.Q_K_p10) / 2.56)
        
        qb_samples = np.maximum(0.0, self.rng.normal(state.Q_B_hat, sigma_qb, self.num_rollout_samples))
        qk_samples = np.maximum(0.0, self.rng.normal(state.Q_K_hat, sigma_qk, self.num_rollout_samples))
        
        regime = state.most_likely_regime.value
        arr_mult = self.config.regime_arrival_multiplier.get(regime, 1.0)
        eff_b = self.config.regime_service_multiplier_billing.get(regime, 1.0)
        eff_k = self.config.regime_service_multiplier_kitchen.get(regime, 1.0)
        
        mu_b = (state.C_B_hat / 0.95) * eff_b
        mu_k = (state.C_K_hat / 3.60) * eff_k
        
        forecasts = []
        
        for k in self.horizons:
            future_t = current_minute_of_day + k
            
            # Estimate base arrival rate from time-of-day profile
            # Multi-peak approximation
            base_rate = 0.8 + 5.0 * np.exp(-((future_t - 290) ** 2) / (2 * 45 ** 2))
            lambda_k = max(0.5, base_rate * arr_mult)
            
            # Rollout transitions across all sample paths
            arr_step = self.rng.poisson(lambda_k, self.num_rollout_samples)
            
            # Billing completions
            serv_b_potential = self.rng.poisson(max(0.1, mu_b), self.num_rollout_samples)
            serv_b_actual = np.minimum(qb_samples + arr_step, serv_b_potential)
            qb_samples = np.maximum(0.0, qb_samples + arr_step - serv_b_actual)
            
            # Kitchen receives completed billing
            serv_k_potential = self.rng.poisson(max(0.1, mu_k), self.num_rollout_samples)
            serv_k_actual = np.minimum(qk_samples + serv_b_actual, serv_k_potential)
            qk_samples = np.maximum(0.0, qk_samples + serv_b_actual - serv_k_actual)
            
            wb_median = np.median(qb_samples) * 0.95
            wk_median = np.median(qk_samples) * 3.60
            
            forecasts.append(HorizonForecast(
                horizon_minutes=k,
                Q_B_p10=round(float(np.percentile(qb_samples, 10)), 2),
                Q_B_median=round(float(np.median(qb_samples)), 2),
                Q_B_p90=round(float(np.percentile(qb_samples, 90)), 2),
                Q_K_p10=round(float(np.percentile(qk_samples, 10)), 2),
                Q_K_median=round(float(np.median(qk_samples)), 2),
                Q_K_p90=round(float(np.percentile(qk_samples, 90)), 2),
                W_B_median=round(float(wb_median), 2),
                W_K_median=round(float(wk_median), 2)
            ))
            
        return forecasts

    @staticmethod
    def calculate_crps(predicted_samples: np.ndarray, ground_truth: float) -> float:
        """
        Continuous Ranked Probability Score (CRPS).
        Evaluates probabilistic forecast quality:
        CRPS = E|X - y| - 0.5 * E|X - X'|
        """
        n = len(predicted_samples)
        if n == 0:
            return 0.0
        term1 = np.mean(np.abs(predicted_samples - ground_truth))
        # Efficient sample pairwise difference
        sorted_samples = np.sort(predicted_samples)
        i = np.arange(1, n + 1)
        term2 = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * sorted_samples)
        return float(term1 - 0.5 * term2)
