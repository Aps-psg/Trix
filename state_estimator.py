"""QueueFlux — Person 2 Core Model: State Estimator
=================================================
Estimates the hidden queue state P(X_t | Y_{1:t}) under incomplete,
noisy, or delayed observations using a Bayesian Particle Filter.

Interfaces:
- Shared Event Format (input):
    {
        "timestamp": 120,
        "event_type": "BILLING_COMPLETED",
        "order_id": "O102",
        "order_type": "MEAL",
        "service_time": 35
    }
- Shared State Format (output):
    {
        "billing_queue": 8,
        "kitchen_queue": 12,
        "billing_workload_min": 6.5,
        "kitchen_workload_min": 18.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 3
    }
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats

from workload_model import OrderType, WorkloadModel


@dataclass
class SharedQueueState:
    """The agreed shared state format across Person 1, Person 2, and Person 3."""
    billing_queue: int
    kitchen_queue: int
    billing_workload_min: float
    kitchen_workload_min: float
    active_billing_counters: int
    active_kitchen_servers: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "billing_queue": int(self.billing_queue),
            "kitchen_queue": int(self.kitchen_queue),
            "billing_workload_min": round(float(self.billing_workload_min), 2),
            "kitchen_workload_min": round(float(self.kitchen_workload_min), 2),
            "active_billing_counters": int(self.active_billing_counters),
            "active_kitchen_servers": int(self.active_kitchen_servers),
        }


@dataclass
class StateEstimateResult:
    """Full probabilistic state estimate containing the agreed shared format

    plus uncertainty credible intervals and diagnostic scores.
    """
    timestamp: float
    state: SharedQueueState
    q_billing_ci_90: Tuple[float, float]      # 90% credible interval [5th, 95th percentile]
    q_kitchen_ci_90: Tuple[float, float]
    w_billing_ci_90: Tuple[float, float]
    w_kitchen_ci_90: Tuple[float, float]
    estimated_arrival_rate: float            # lambda_t (customers/min)
    estimated_billing_rate: float            # mu_t^B (services/min)
    estimated_kitchen_rate: float            # mu_t^K (services/min)
    uncertainty_score: float                 # Normalized dispersion in [0, 1]
    effective_sample_size: float
    last_update_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        base = self.state.to_dict()
        base.update({
            "timestamp": self.timestamp,
            "q_billing_ci_90": [round(self.q_billing_ci_90[0], 1), round(self.q_billing_ci_90[1], 1)],
            "q_kitchen_ci_90": [round(self.q_kitchen_ci_90[0], 1), round(self.q_kitchen_ci_90[1], 1)],
            "w_billing_ci_90": [round(self.w_billing_ci_90[0], 2), round(self.w_billing_ci_90[1], 2)],
            "w_kitchen_ci_90": [round(self.w_kitchen_ci_90[0], 2), round(self.w_kitchen_ci_90[1], 2)],
            "estimated_arrival_rate": round(self.estimated_arrival_rate, 2),
            "estimated_billing_rate": round(self.estimated_billing_rate, 2),
            "estimated_kitchen_rate": round(self.estimated_kitchen_rate, 2),
            "uncertainty_score": round(self.uncertainty_score, 3),
            "effective_sample_size": round(self.effective_sample_size, 1),
        })
        return base


class ParticleFilterQueueEstimator:
    """Bayesian Particle Filter estimating hidden queue counts and workloads

    under noisy camera reports and incomplete event transmissions.
    """

    def __init__(
        self,
        num_particles: int = 150,
        active_billing_counters: int = 2,
        active_kitchen_servers: int = 3,
        seed: Optional[int] = 42,
    ) -> None:
        self.num_particles = num_particles
        self.active_billing_counters = active_billing_counters
        self.active_kitchen_servers = active_kitchen_servers
        self.workload_model = WorkloadModel()
        self.rng = np.random.default_rng(seed)

        # Particle states: (q_b, q_k, w_b, w_k, lambda)
        self.particles_qb = self.rng.integers(0, 3, size=num_particles).astype(float)
        self.particles_qk = self.rng.integers(0, 3, size=num_particles).astype(float)
        self.particles_wb = self.particles_qb * self.workload_model.billing_params.mean_minutes
        self.particles_wk = self.particles_qk * self.workload_model.get_expected_kitchen_order_duration()
        self.particles_lambda = np.full(num_particles, 1.8)
        self.weights = np.ones(num_particles) / num_particles

        # Smoothed rate parameters
        self.est_billing_service_rate = 1.33  # ~0.75 min/service
        self.est_kitchen_service_rate = 0.33  # ~3.0 min/service
        self.current_time = 0.0

    def predict(self, delta_time_minutes: float = 1.0) -> None:
        """Propagation step: project particles forward using queue balance dynamics:

        Q_{t+dt}^B = max(0, Q_t^B + A_{dt} - S_{dt}^B)
        Q_{t+dt}^K = max(0, Q_t^K + S_{dt}^B - S_{dt}^K)
        """
        dt = max(0.01, float(delta_time_minutes))
        self.current_time += dt

        # 1. Sample new arrivals: Poisson(lambda * dt)
        arrivals = self.rng.poisson(self.particles_lambda * dt).astype(float)

        # 2. Sample billing service completions: Poisson(C^B * mu^B * dt)
        cap_b = self.rng.poisson(
            self.active_billing_counters * self.est_billing_service_rate * dt,
            size=self.num_particles,
        ).astype(float)
        service_b = np.minimum(self.particles_qb + arrivals, cap_b)

        # 3. Update billing queue
        self.particles_qb = np.maximum(0.0, self.particles_qb + arrivals - service_b)
        self.particles_wb = self.particles_qb * self.workload_model.billing_params.mean_minutes

        # 4. Billed orders enter kitchen queue: E_t = S_t^B
        cap_k = self.rng.poisson(
            self.active_kitchen_servers * self.est_kitchen_service_rate * dt,
            size=self.num_particles,
        ).astype(float)
        service_k = np.minimum(self.particles_qk + service_b, cap_k)

        # 5. Update kitchen queue
        self.particles_qk = np.maximum(0.0, self.particles_qk + service_b - service_k)
        avg_k_dur = self.workload_model.get_expected_kitchen_order_duration()
        self.particles_wk = self.particles_qk * avg_k_dur

    def update_from_events(
        self,
        events: List[Dict[str, Any]],
        noisy_camera_billing_count: Optional[int] = None,
        detection_probability: float = 0.85,
    ) -> StateEstimateResult:
        """Observation update: Incorporate timestamped events and sensor observations.

        Supports standard shared event format:
        [
            {"timestamp": 120, "event_type": "BILLING_COMPLETED", "order_id": "O102", "order_type": "MEAL", ...}
        ]
        """
        # Count event types in this interval
        arrived_events = sum(1 for e in events if e.get("event_type") in ("ARRIVED", "ARRIVAL"))
        billing_done = sum(1 for e in events if e.get("event_type") in ("BILLING_COMPLETED", "PAID"))
        kitchen_done = sum(1 for e in events if e.get("event_type") in ("KITCHEN_COMPLETED", "READY", "COLLECTED"))

        # Update observed arrival rate
        if arrived_events > 0:
            self.particles_lambda = 0.8 * self.particles_lambda + 0.2 * float(arrived_events)

        # Compute observation likelihood
        likelihoods = np.ones(self.num_particles)

        # 1. Camera count likelihood: Y_cam ~ Normal(Q_B, sigma=1.5)
        if noisy_camera_billing_count is not None:
            cam_val = float(noisy_camera_billing_count)
            cam_lik = stats.norm.pdf(cam_val, loc=self.particles_qb, scale=1.5)
            likelihoods *= np.maximum(cam_lik, 1e-5)

        # 2. Billing completion likelihood: Poisson approximation
        if billing_done > 0:
            exp_b_obs = (self.particles_qb > 0).astype(float) * self.active_billing_counters * self.est_billing_service_rate * detection_probability
            b_lik = stats.poisson.pmf(billing_done, mu=np.maximum(0.1, exp_b_obs))
            likelihoods *= np.maximum(b_lik, 1e-5)

        # 3. Kitchen completion likelihood
        if kitchen_done > 0:
            exp_k_obs = (self.particles_qk > 0).astype(float) * self.active_kitchen_servers * self.est_kitchen_service_rate * detection_probability
            k_lik = stats.poisson.pmf(kitchen_done, mu=np.maximum(0.1, exp_k_obs))
            likelihoods *= np.maximum(k_lik, 1e-5)

        # Weight update and normalization
        self.weights *= likelihoods
        sum_w = np.sum(self.weights)
        if sum_w <= 0.0 or np.isnan(sum_w):
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights /= sum_w

        # Resample if effective sample size drops: N_eff < N / 2
        n_eff = 1.0 / float(np.sum(self.weights ** 2))
        if n_eff < (self.num_particles / 2.0):
            indices = self.rng.choice(self.num_particles, size=self.num_particles, p=self.weights)
            self.particles_qb = self.particles_qb[indices]
            self.particles_qk = self.particles_qk[indices]
            self.particles_wb = self.particles_wb[indices]
            self.particles_wk = self.particles_wk[indices]
            self.particles_lambda = self.particles_lambda[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
            n_eff = float(self.num_particles)

        # Extract posterior summary
        exp_qb = float(np.average(self.particles_qb, weights=self.weights))
        exp_qk = float(np.average(self.particles_qk, weights=self.weights))
        exp_wb = float(np.average(self.particles_wb, weights=self.weights))
        exp_wk = float(np.average(self.particles_wk, weights=self.weights))

        # 90% Credible Intervals [5th, 95th percentiles]
        sorted_qb = np.sort(self.particles_qb)
        sorted_qk = np.sort(self.particles_qk)
        sorted_wb = np.sort(self.particles_wb)
        sorted_wk = np.sort(self.particles_wk)

        ci_qb = (float(sorted_qb[int(0.05 * self.num_particles)]), float(sorted_qb[int(0.95 * self.num_particles)]))
        ci_qk = (float(sorted_qk[int(0.05 * self.num_particles)]), float(sorted_qk[int(0.95 * self.num_particles)]))
        ci_wb = (float(sorted_wb[int(0.05 * self.num_particles)]), float(sorted_wb[int(0.95 * self.num_particles)]))
        ci_wk = (float(sorted_wk[int(0.05 * self.num_particles)]), float(sorted_wk[int(0.95 * self.num_particles)]))

        # Uncertainty score in [0, 1]
        std_qb = float(np.std(self.particles_qb))
        uncertainty = min(1.0, std_qb / max(2.0, exp_qb + 1.0))

        shared_state = SharedQueueState(
            billing_queue=int(round(exp_qb)),
            kitchen_queue=int(round(exp_qk)),
            billing_workload_min=round(exp_wb, 2),
            kitchen_workload_min=round(exp_wk, 2),
            active_billing_counters=self.active_billing_counters,
            active_kitchen_servers=self.active_kitchen_servers,
        )

        return StateEstimateResult(
            timestamp=self.current_time,
            state=shared_state,
            q_billing_ci_90=(round(ci_qb[0], 1), round(ci_qb[1], 1)),
            q_kitchen_ci_90=(round(ci_qk[0], 1), round(ci_qk[1], 1)),
            w_billing_ci_90=(round(ci_wb[0], 2), round(ci_wb[1], 2)),
            w_kitchen_ci_90=(round(ci_wk[0], 2), round(ci_wk[1], 2)),
            estimated_arrival_rate=round(float(np.mean(self.particles_lambda)), 2),
            estimated_billing_rate=round(self.est_billing_service_rate, 2),
            estimated_kitchen_rate=round(self.est_kitchen_service_rate, 2),
            uncertainty_score=round(uncertainty, 3),
            effective_sample_size=round(n_eff, 1),
            last_update_timestamp=self.current_time,
        )


class EventCountBaselineEstimator:
    """Baseline Estimator: Simple naive bookkeeping without Bayesian filtering."""

    def __init__(self, active_billing: int = 2, active_kitchen: int = 3) -> None:
        self.qb = 0
        self.qk = 0
        self.active_billing = active_billing
        self.active_kitchen = active_kitchen
        self.workload_model = WorkloadModel()

    def process_events(self, events: List[Dict[str, Any]], noisy_camera: Optional[int] = None) -> SharedQueueState:
        for e in events:
            etype = e.get("event_type", "")
            if etype in ("ARRIVED", "ARRIVAL"):
                self.qb += 1
            elif etype in ("BILLING_COMPLETED", "PAID"):
                self.qb = max(0, self.qb - 1)
                self.qk += 1
            elif etype in ("KITCHEN_COMPLETED", "READY", "COLLECTED"):
                self.qk = max(0, self.qk - 1)

        # If camera count is present, directly overwrite naive billing count
        if noisy_camera is not None:
            self.qb = max(0, noisy_camera)

        wb = self.workload_model.estimate_stage_workload(self.qb, stage="billing")
        wk = self.workload_model.estimate_stage_workload(self.qk, stage="kitchen")

        return SharedQueueState(
            billing_queue=self.qb,
            kitchen_queue=self.qk,
            billing_workload_min=wb,
            kitchen_workload_min=wk,
            active_billing_counters=self.active_billing,
            active_kitchen_servers=self.active_kitchen,
        )


def generate_mock_events(start_t_minutes: float, duration_minutes: float = 1.0, count: int = 4) -> List[Dict[str, Any]]:
    """Helper for Person 2: Generate mock events adhering to the shared event format

    so state estimation can be tested without waiting for Person 1.
    """
    event_types = ["ARRIVED", "BILLING_COMPLETED", "KITCHEN_COMPLETED"]
    order_types = ["BEVERAGE", "SNACK", "MEAL", "COMBO"]
    rng = np.random.default_rng(123)

    mock_events = []
    for i in range(count):
        t_sec = int((start_t_minutes + rng.uniform(0.0, duration_minutes)) * 60)
        mock_events.append({
            "timestamp": t_sec,
            "event_type": str(rng.choice(event_types)),
            "order_id": f"MOCK-{i+1:03d}",
            "order_type": str(rng.choice(order_types)),
            "service_time": int(rng.integers(20, 60)),
        })
    return mock_events
