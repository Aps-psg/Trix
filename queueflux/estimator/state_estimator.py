"""
Queue State Estimator for QueueFlux under incomplete and noisy observations.
Fully vectorized Sequential Importance Resampling (SIR) Particle Filter coupled
with a Discrete-State Hidden Markov Model (HMM) for operating regime inference.
Guarantees zero leakage of ground-truth state during inference.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from queueflux.config import SystemConfig, DEFAULT_CONFIG
from queueflux.models.entities import RegimeType
from queueflux.models.state import ObservationVector, EstimatedState


class QueueStateEstimator:
    """
    High-performance vectorized Particle-Filter Bayesian Observer for QueueFlux.
    Maintains and updates the belief distribution p(X_t | Y_{1:t}) over unobserved
    queue lengths, remaining workloads, and hidden operating regimes.
    """

    def __init__(self, num_particles: int = 500, config: SystemConfig = DEFAULT_CONFIG, seed: int = 42):
        self.num_particles = num_particles
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Nominal service durations (minutes)
        self.avg_billing_service = 0.95
        self.avg_kitchen_workload = 3.60
        
        # Regimes: 0=NORMAL, 1=SURGE, 2=SLOWDOWN, 3=BOTTLENECK
        self.regimes = [RegimeType.NORMAL.value, RegimeType.SURGE.value, 
                        RegimeType.SLOWDOWN.value, RegimeType.BOTTLENECK.value]
        self.trans_matrix = np.array([
            [0.92, 0.04, 0.02, 0.02],   # NORMAL ->
            [0.10, 0.85, 0.02, 0.03],   # SURGE ->
            [0.08, 0.02, 0.85, 0.05],   # SLOWDOWN ->
            [0.08, 0.02, 0.02, 0.88]    # BOTTLENECK ->
        ])
        self.trans_cumsum = np.cumsum(self.trans_matrix, axis=1)

        # Regimes multipliers arrays: shape (4,)
        self.arr_mults = np.array([1.0, 2.2, 1.0, 1.1])
        self.eff_b_mults = np.array([1.0, 0.95, 0.45, 0.90])
        self.eff_k_mults = np.array([1.0, 0.90, 0.85, 0.40])
        
        # Vectorized Particle State Arrays: shape (N,)
        self.Q_B = np.zeros(num_particles, dtype=np.float64)
        self.Q_K = np.zeros(num_particles, dtype=np.float64)
        self.W_B = np.zeros(num_particles, dtype=np.float64)
        self.W_K = np.zeros(num_particles, dtype=np.float64)
        self.residual_B = np.zeros(num_particles, dtype=np.float64)
        self.residual_K = np.zeros(num_particles, dtype=np.float64)
        self.regime_indices = np.zeros(num_particles, dtype=np.int32)
        self.weights = np.ones(num_particles, dtype=np.float64) / num_particles
        
        self._initialize_particles()

    def _initialize_particles(self):
        """Initializes particles representing the morning start (near-empty system)."""
        n = self.num_particles
        self.Q_B = np.maximum(0.0, self.rng.normal(0.0, 0.5, n))
        self.Q_K = np.maximum(0.0, self.rng.normal(0.0, 0.5, n))
        self.W_B = self.Q_B * self.avg_billing_service
        self.W_K = self.Q_K * self.avg_kitchen_workload
        self.residual_B = np.zeros(n)
        self.residual_K = np.zeros(n)
        self.regime_indices = np.zeros(n, dtype=np.int32)  # NORMAL
        self.weights = np.ones(n, dtype=np.float64) / n

    def update(self, obs: ObservationVector) -> EstimatedState:
        """
        Vectorized Bayesian update given observation Y_t:
        1. Predict: propagate particles through system transition dynamics
        2. Measurement update: compute likelihoods and re-weight particles
        3. Systematic resample if effective sample size is degraded
        4. Extract posterior belief moments and credible intervals
        """
        n = self.num_particles
        c_b = float(obs.c_B_reported)
        c_k = float(obs.c_K_reported)
        
        # --- 1. PREDICTION STEP ---
        # A. Vectorized Markov regime transition
        u = self.rng.uniform(0, 1, n)
        # Select next regime using row corresponding to current regime
        cum_probs = self.trans_cumsum[self.regime_indices]  # (n, 4)
        self.regime_indices = (u[:, None] > cum_probs).sum(axis=1)
        self.regime_indices = np.clip(self.regime_indices, 0, 3)

        # Multipliers
        arr_m = self.arr_mults[self.regime_indices]
        eff_b = self.eff_b_mults[self.regime_indices]
        eff_k = self.eff_k_mults[self.regime_indices]

        # Inferred Poisson arrival intensity
        ambient_factor = max(0.5, (obs.ambient_noise_level - 55.0) / 15.0)
        lambda_arr = np.maximum(0.4, 2.2 * ambient_factor * arr_m)
        a_samples = self.rng.poisson(lambda_arr)

        # Billing service potential
        mu_b_cap = np.maximum(0.1, (c_b / self.avg_billing_service) * eff_b)
        s_b_potential = self.rng.poisson(mu_b_cap)
        s_b_actual = np.minimum(self.Q_B + a_samples, s_b_potential)

        self.Q_B = np.maximum(0.0, self.Q_B + a_samples - s_b_actual)
        self.W_B = np.maximum(0.0, self.Q_B * self.avg_billing_service + self.rng.normal(0, 0.15, n))
        self.residual_B = np.maximum(0.0, np.minimum(c_b * 0.4, self.residual_B * 0.5 + self.rng.uniform(0, 0.4, n)))

        # Kitchen dynamics (orders entering kitchen = s_b_actual)
        e_samples = s_b_actual
        mu_k_cap = np.maximum(0.1, (c_k / self.avg_kitchen_workload) * eff_k)
        s_k_potential = self.rng.poisson(mu_k_cap)
        s_k_actual = np.minimum(self.Q_K + e_samples, s_k_potential)

        self.Q_K = np.maximum(0.0, self.Q_K + e_samples - s_k_actual)
        self.W_K = np.maximum(0.0, self.Q_K * self.avg_kitchen_workload + self.rng.normal(0, 0.25, n))
        self.residual_K = np.maximum(0.0, np.minimum(c_k * 1.5, self.residual_K * 0.5 + self.rng.uniform(0, 0.8, n)))

        # --- 2. MEASUREMENT UPDATE STEP (Log-space with Log-Sum-Exp) ---
        sig_b = max(0.8, self.config.camera_count_noise_std)
        sig_k = max(0.8, self.config.kitchen_sensor_noise_std)

        diff_b = obs.y_Q_B - self.Q_B
        diff_k = obs.y_Q_K - self.Q_K

        # Visual camera log-likelihood (accounting for camera noise and occlusion)
        log_lik_b = -0.5 * (diff_b / sig_b) ** 2
        log_lik_k = -0.5 * (diff_k / sig_k) ** 2
        step_log_lik = log_lik_b + log_lik_k

        # Stabilized log-sum-exp particle re-weighting
        log_w = np.log(np.maximum(1e-12, self.weights)) + step_log_lik
        log_w -= np.max(log_w)
        unnorm_w = np.exp(log_w)
        self.weights = unnorm_w / np.sum(unnorm_w)

        # --- 3. RESAMPLING & ADAPTIVE REJUVENATION STEP ---
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < n * 0.8:
            self._vectorized_systematic_resample()

        # Rejuvenation: If particles lose track of sensor observations (outlier or unmodeled shock)
        median_qb = np.median(self.Q_B)
        median_qk = np.median(self.Q_K)
        if abs(median_qb - obs.y_Q_B) > 4.0 * sig_b:
            # Re-seed 25% of particles around sensor observation
            reseed_idx = self.rng.choice(n, size=int(n * 0.25), replace=False)
            self.Q_B[reseed_idx] = np.maximum(0.0, obs.y_Q_B + self.rng.normal(0, sig_b, len(reseed_idx)))
            self.W_B[reseed_idx] = self.Q_B[reseed_idx] * self.avg_billing_service

        if abs(median_qk - obs.y_Q_K) > 4.0 * sig_k:
            reseed_idx = self.rng.choice(n, size=int(n * 0.25), replace=False)
            self.Q_K[reseed_idx] = np.maximum(0.0, obs.y_Q_K + self.rng.normal(0, sig_k, len(reseed_idx)))
            self.W_K[reseed_idx] = self.Q_K[reseed_idx] * self.avg_kitchen_workload

        # --- 4. EXTRACT POSTERIOR BELIEF MOMENTS ---
        regime_counts = np.bincount(self.regime_indices, weights=self.weights, minlength=4)
        regime_probs = {
            self.regimes[i]: round(float(regime_counts[i] / np.sum(regime_counts)), 4)
            for i in range(4)
        }

        return EstimatedState(
            time_t=obs.time_t,
            Q_B_hat=float(np.median(self.Q_B)),
            Q_B_p10=float(np.percentile(self.Q_B, 10)),
            Q_B_p90=float(np.percentile(self.Q_B, 90)),
            Q_K_hat=float(np.median(self.Q_K)),
            Q_K_p10=float(np.percentile(self.Q_K, 10)),
            Q_K_p90=float(np.percentile(self.Q_K, 90)),
            W_B_hat=float(np.median(self.W_B)),
            W_B_p10=float(np.percentile(self.W_B, 10)),
            W_B_p90=float(np.percentile(self.W_B, 90)),
            W_K_hat=float(np.median(self.W_K)),
            W_K_p10=float(np.percentile(self.W_K, 10)),
            W_K_p90=float(np.percentile(self.W_K, 90)),
            residual_B_hat=float(np.mean(self.residual_B)),
            residual_K_hat=float(np.mean(self.residual_K)),
            C_B_hat=c_b,
            C_K_hat=c_k,
            regime_probs=regime_probs
        )

    def _vectorized_systematic_resample(self):
        """Vectorized O(N) systematic resampling with diversity preservation."""
        n = self.num_particles
        positions = (self.rng.uniform(0, 1) + np.arange(n)) / n
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        indices = np.clip(np.searchsorted(cumulative_sum, positions), 0, n - 1)

        # Resample arrays
        self.Q_B = np.maximum(0.0, self.Q_B[indices] + self.rng.normal(0, 0.12, n))
        self.Q_K = np.maximum(0.0, self.Q_K[indices] + self.rng.normal(0, 0.18, n))
        self.W_B = np.maximum(0.0, self.W_B[indices] + self.rng.normal(0, 0.12, n))
        self.W_K = np.maximum(0.0, self.W_K[indices] + self.rng.normal(0, 0.22, n))
        self.residual_B = np.maximum(0.0, self.residual_B[indices])
        self.residual_K = np.maximum(0.0, self.residual_K[indices])
        self.regime_indices = self.regime_indices[indices]
        self.weights = np.ones(n, dtype=np.float64) / n
