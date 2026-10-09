"""QueueFlux — Person 2 Core Model: Wait Predictor
=================================================
Workload-aware waiting-time predictor for registration-to-collection latency:
    hat{T}_{total} = hat{T}_{wait}^B + S^B + hat{T}_{wait}^K + S^K(order_type)

Accounts for:
- Workload ahead (in expected service-minutes)
- Renewal-theory residual service times of active servers
- Order-type specific preparation requirements
- Uncertainty intervals (10th, 50th/median, 90th percentiles)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from state_estimator import SharedQueueState, StateEstimateResult
from workload_model import OrderType, WorkloadModel


@dataclass
class WaitingTimePrediction:
    """Predicted waiting time distribution from registration to collection."""
    order_type: str
    expected_total_wait_min: float
    median_total_wait_min: float
    lower_bound_10_min: float
    upper_bound_90_min: float
    stage_a_billing_min: float
    stage_b_kitchen_min: float
    billing_wait_only_min: float
    kitchen_wait_only_min: float
    order_prep_time_min: float
    exceedance_probability_15min: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_type": self.order_type,
            "expected_total_wait_min": round(self.expected_total_wait_min, 2),
            "median_total_wait_min": round(self.median_total_wait_min, 2),
            "lower_bound_10_min": round(self.lower_bound_10_min, 2),
            "upper_bound_90_min": round(self.upper_bound_90_min, 2),
            "stage_a_billing_min": round(self.stage_a_billing_min, 2),
            "stage_b_kitchen_min": round(self.stage_b_kitchen_min, 2),
            "billing_wait_only_min": round(self.billing_wait_only_min, 2),
            "kitchen_wait_only_min": round(self.kitchen_wait_only_min, 2),
            "order_prep_time_min": round(self.order_prep_time_min, 2),
            "exceedance_probability_15min": round(self.exceedance_probability_15min, 3),
        }


class WorkloadWaitPredictor:
    """Predicts registration-to-collection waiting times using workload service-minutes

    and renewal residual service times across active servers.
    """

    def __init__(self, workload_model: Optional[WorkloadModel] = None) -> None:
        self.workload_model = workload_model or WorkloadModel()

    def predict_waiting_time(
        self,
        order_type: Union[OrderType, str],
        state_or_estimate: Union[SharedQueueState, StateEstimateResult, Dict[str, Any]],
        active_billing_counters: Optional[int] = None,
        active_kitchen_servers: Optional[int] = None,
        threshold_minutes: float = 15.0,
    ) -> WaitingTimePrediction:
        """Compute expected total waiting time and prediction interval for an order."""
        ot = order_type if isinstance(order_type, OrderType) else OrderType.from_str(str(order_type))

        # Extract state variables
        if isinstance(state_or_estimate, StateEstimateResult):
            wb = state_or_estimate.state.billing_workload_min
            wk = state_or_estimate.state.kitchen_workload_min
            c_b = state_or_estimate.state.active_billing_counters
            c_k = state_or_estimate.state.active_kitchen_servers
            wb_ci = state_or_estimate.w_billing_ci_90
            wk_ci = state_or_estimate.w_kitchen_ci_90
        elif isinstance(state_or_estimate, SharedQueueState):
            wb = state_or_estimate.billing_workload_min
            wk = state_or_estimate.kitchen_workload_min
            c_b = state_or_estimate.active_billing_counters
            c_k = state_or_estimate.active_kitchen_servers
            wb_ci = (wb * 0.7, wb * 1.3)
            wk_ci = (wk * 0.7, wk * 1.3)
        else:
            wb = float(state_or_estimate.get("billing_workload_min", 0.0))
            wk = float(state_or_estimate.get("kitchen_workload_min", 0.0))
            c_b = int(state_or_estimate.get("active_billing_counters", 2))
            c_k = int(state_or_estimate.get("active_kitchen_servers", 3))
            wb_ci = (wb * 0.7, wb * 1.3)
            wk_ci = (wk * 0.7, wk * 1.3)

        # Allow explicit override of server capacities
        if active_billing_counters is not None:
            c_b = max(1, active_billing_counters)
        else:
            c_b = max(1, c_b)

        if active_kitchen_servers is not None:
            c_k = max(1, active_kitchen_servers)
        else:
            c_k = max(1, c_k)

        # 1. Stage A: Billing Wait Time
        # Workload ahead + renewal residual service times of currently active servers
        residual_b = self.workload_model.get_residual_service_workload(active_servers=c_b, stage="billing")
        t_wait_billing = (wb + residual_b) / float(c_b)
        t_service_billing = self.workload_model.billing_params.mean_minutes
        total_billing_stage = t_wait_billing + t_service_billing

        # 2. Stage B: Kitchen Wait Time
        residual_k = self.workload_model.get_residual_service_workload(active_servers=c_k, stage="kitchen")
        t_wait_kitchen = (wk + residual_k) / float(c_k)
        t_service_kitchen = self.workload_model.get_order_expected_workload(ot, stage="kitchen")
        total_kitchen_stage = t_wait_kitchen + t_service_kitchen

        expected_total = total_billing_stage + total_kitchen_stage

        # 3. Uncertainty Intervals from Credible Intervals
        lower_b = (wb_ci[0] + residual_b * 0.5) / float(c_b) + (t_service_billing * 0.75)
        upper_b = (wb_ci[1] + residual_b * 1.5) / float(c_b) + (t_service_billing * 1.25)

        lower_k = (wk_ci[0] + residual_k * 0.5) / float(c_k) + (t_service_kitchen * 0.75)
        upper_k = (wk_ci[1] + residual_k * 1.5) / float(c_k) + (t_service_kitchen * 1.25)

        lower_bound = max(t_service_billing + t_service_kitchen, lower_b + lower_k)
        upper_bound = max(expected_total * 1.05, upper_b + upper_k)
        median_total = 0.5 * (expected_total + 0.5 * (lower_bound + upper_bound))

        # 4. Probability of exceeding threshold (assuming normal approximation of log-durations)
        std_est = max(0.5, (upper_bound - lower_bound) / 3.29)
        z = (threshold_minutes - expected_total) / std_est
        # P(T > threshold) = 1 - Phi(z)
        exceed_prob = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
        exceed_prob = max(0.0, min(1.0, exceed_prob))

        return WaitingTimePrediction(
            order_type=ot.value,
            expected_total_wait_min=round(expected_total, 2),
            median_total_wait_min=round(median_total, 2),
            lower_bound_10_min=round(lower_bound, 2),
            upper_bound_90_min=round(upper_bound, 2),
            stage_a_billing_min=round(total_billing_stage, 2),
            stage_b_kitchen_min=round(total_kitchen_stage, 2),
            billing_wait_only_min=round(t_wait_billing, 2),
            kitchen_wait_only_min=round(t_wait_kitchen, 2),
            order_prep_time_min=round(t_service_kitchen, 2),
            exceedance_probability_15min=round(exceed_prob, 3),
        )

    def predict_all_order_types(
        self,
        state_or_estimate: Union[SharedQueueState, StateEstimateResult, Dict[str, Any]],
        active_billing_counters: Optional[int] = None,
        active_kitchen_servers: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Batch helper predicting waiting times for all 4 standard order types."""
        results: Dict[str, Dict[str, Any]] = {}
        for ot in OrderType:
            pred = self.predict_waiting_time(
                order_type=ot,
                state_or_estimate=state_or_estimate,
                active_billing_counters=active_billing_counters,
                active_kitchen_servers=active_kitchen_servers,
            )
            results[ot.value] = pred.to_dict()
        return results
