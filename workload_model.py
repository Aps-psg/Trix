"""QueueFlux — Person 2 Core Model: Workload Model
=================================================
Calculates workload in consistent units (expected service-minutes),
computes renewal-theory residual service times for active servers,
and models positive-valued service-time distributions (lognormal/gamma).

Units:
- Workload: expected service-minutes
- Timestamps: minutes (standard) / seconds (auto-converted)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class OrderType(str, Enum):
    BEVERAGE = "BEVERAGE"  # Fast: mean ~ 1.2 min
    SNACK = "SNACK"        # Medium: mean ~ 2.5 min
    MEAL = "MEAL"          # Standard: mean ~ 5.0 min
    COMBO = "COMBO"        # Heavy: mean ~ 7.5 min

    @classmethod
    def from_str(cls, val: str) -> OrderType:
        normalized = val.strip().upper()
        for member in cls:
            if member.value == normalized:
                return member
        return cls.MEAL  # Safe fallback default


@dataclass(frozen=True)
class ServiceDistributionParams:
    """Hyperparameters for lognormal service-time distribution."""
    mean_minutes: float
    sigma: float  # log-normal shape parameter

    @property
    def mu(self) -> float:
        """Calculate log-normal location parameter mu from mean and sigma."""
        return math.log(max(1e-4, self.mean_minutes)) - (self.sigma ** 2) / 2.0

    @property
    def variance(self) -> float:
        """Calculate variance of the lognormal distribution."""
        return (math.exp(self.sigma ** 2) - 1.0) * (self.mean_minutes ** 2)

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance)

    @property
    def cv(self) -> float:
        """Coefficient of variation: sigma / mean."""
        return self.std_dev / self.mean_minutes

    @property
    def mean_residual_service_time(self) -> float:
        """Renewal theory expected residual/excess life:

        E[R] = E[S^2] / (2 * E[S]) = (mean / 2) * (1 + CV^2)
        """
        return (self.mean_minutes / 2.0) * (1.0 + self.cv ** 2)


# Stage A (Billing) parameters: mean 45 seconds = 0.75 minutes
DEFAULT_BILLING_PARAMS = ServiceDistributionParams(mean_minutes=0.75, sigma=0.25)

# Stage B (Kitchen) parameters by order type
DEFAULT_KITCHEN_PARAMS: Dict[OrderType, ServiceDistributionParams] = {
    OrderType.BEVERAGE: ServiceDistributionParams(mean_minutes=1.2, sigma=0.25),
    OrderType.SNACK: ServiceDistributionParams(mean_minutes=2.5, sigma=0.28),
    OrderType.MEAL: ServiceDistributionParams(mean_minutes=5.0, sigma=0.30),
    OrderType.COMBO: ServiceDistributionParams(mean_minutes=7.5, sigma=0.32),
}

# Default empirical order mix in a college canteen
DEFAULT_ORDER_MIX: Dict[OrderType, float] = {
    OrderType.BEVERAGE: 0.35,
    OrderType.SNACK: 0.30,
    OrderType.MEAL: 0.25,
    OrderType.COMBO: 0.10,
}


class WorkloadModel:
    """Workload evaluation engine translating queue counts and order types

    into expected service-minutes.
    """

    def __init__(
        self,
        billing_params: Optional[ServiceDistributionParams] = None,
        kitchen_params: Optional[Dict[OrderType, ServiceDistributionParams]] = None,
        order_mix: Optional[Dict[OrderType, float]] = None,
    ) -> None:
        self.billing_params = billing_params or DEFAULT_BILLING_PARAMS
        self.kitchen_params = kitchen_params or DEFAULT_KITCHEN_PARAMS
        self.order_mix = order_mix or DEFAULT_ORDER_MIX

    def get_order_expected_workload(self, order_type: Union[OrderType, str], stage: str = "kitchen") -> float:
        """Return expected service workload in minutes for a specific order and stage."""
        stage_norm = stage.lower().strip()
        if stage_norm in ("billing", "stage_a"):
            return self.billing_params.mean_minutes

        ot = order_type if isinstance(order_type, OrderType) else OrderType.from_str(order_type)
        params = self.kitchen_params.get(ot, self.kitchen_params[OrderType.MEAL])
        return params.mean_minutes

    def get_order_workload_variance(self, order_type: Union[OrderType, str], stage: str = "kitchen") -> float:
        """Return service workload variance in minutes^2."""
        stage_norm = stage.lower().strip()
        if stage_norm in ("billing", "stage_a"):
            return self.billing_params.variance

        ot = order_type if isinstance(order_type, OrderType) else OrderType.from_str(order_type)
        params = self.kitchen_params.get(ot, self.kitchen_params[OrderType.MEAL])
        return params.variance

    def get_expected_kitchen_order_duration(self, custom_mix: Optional[Dict[Union[OrderType, str], float]] = None) -> float:
        """Compute the weighted-average kitchen preparation duration across all order types."""
        mix = custom_mix or self.order_mix
        weighted_sum = 0.0
        total_weight = 0.0
        for ot_key, weight in mix.items():
            ot = ot_key if isinstance(ot_key, OrderType) else OrderType.from_str(str(ot_key))
            duration = self.get_order_expected_workload(ot, stage="kitchen")
            weighted_sum += duration * weight
            total_weight += weight
        return weighted_sum / max(1e-6, total_weight)

    def estimate_stage_workload(
        self,
        queue_count: int,
        stage: str = "kitchen",
        order_mix: Optional[Dict[Union[OrderType, str], float]] = None,
    ) -> float:
        """Estimate total waiting workload ahead in expected service-minutes:

        W_ahead = sum_{i in Queue} E[S_i]
        """
        if queue_count <= 0:
            return 0.0

        stage_norm = stage.lower().strip()
        if stage_norm in ("billing", "stage_a"):
            return round(queue_count * self.billing_params.mean_minutes, 3)

        avg_kitchen_duration = self.get_expected_kitchen_order_duration(custom_mix=order_mix)
        return round(queue_count * avg_kitchen_duration, 3)

    def calculate_workload_from_orders(self, orders: List[Dict[str, Any]], stage: str = "kitchen") -> float:
        """Compute exact summed workload from a list of specific order objects."""
        if not orders:
            return 0.0

        total_workload = 0.0
        for o in orders:
            # Check for explicit workload or derive from order_type
            if "workload_min" in o:
                total_workload += float(o["workload_min"])
            elif "service_time" in o and o["service_time"] is not None:
                st = float(o["service_time"])
                # Auto-convert if service_time is provided in seconds (> 15)
                st_min = st / 60.0 if st > 15.0 else st
                total_workload += st_min
            else:
                ot_str = o.get("order_type", o.get("category", "MEAL"))
                total_workload += self.get_order_expected_workload(ot_str, stage=stage)

        return round(total_workload, 3)

    def get_residual_service_workload(
        self,
        active_servers: int,
        stage: str = "billing",
    ) -> float:
        """Estimate the remaining (residual) workload currently in progress across active servers.

        Based on renewal theory: E[R] = (mean / 2) * (1 + CV^2).
        Total active residual workload = active_servers * E[R].
        """
        if active_servers <= 0:
            return 0.0

        stage_norm = stage.lower().strip()
        if stage_norm in ("billing", "stage_a"):
            res_per_server = self.billing_params.mean_residual_service_time
        else:
            # Weighted average residual time for kitchen
            res_per_server = sum(
                params.mean_residual_service_time * self.order_mix.get(ot, 0.25)
                for ot, params in self.kitchen_params.items()
            )

        return round(active_servers * res_per_server, 3)

    def sample_service_duration(
        self,
        order_type: Union[OrderType, str],
        stage: str = "kitchen",
        slowdown_multiplier: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Sample a positive stochastic service duration from the lognormal distribution."""
        local_rng = rng if rng is not None else np.random.default_rng()
        stage_norm = stage.lower().strip()

        if stage_norm in ("billing", "stage_a"):
            params = self.billing_params
        else:
            ot = order_type if isinstance(order_type, OrderType) else OrderType.from_str(order_type)
            params = self.kitchen_params.get(ot, self.kitchen_params[OrderType.MEAL])

        raw_duration = float(local_rng.lognormal(params.mu, params.sigma))
        duration = max(0.1, raw_duration * slowdown_multiplier)
        return round(duration, 3)
