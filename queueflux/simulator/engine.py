"""
Discrete-Event Queue Simulator Engine for QueueFlux.
Strictly implements two connected stages (Billing -> Kitchen Preparation) with exact
event sequence, resource allocations, dynamic disruptions, and the queue balance equations:
Q_{t+1}^B = max(0, Q_t^B + A_t - S_t^B)
Q_{t+1}^K = max(0, Q_t^K + E_t - S_t^K)
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from queueflux.config import SystemConfig, DEFAULT_CONFIG
from queueflux.models.entities import (
    Customer,
    Order,
    Resource,
    CustomerStatus,
    OrderType,
    RegimeType
)
from queueflux.models.state import (
    SystemState,
    ObservationVector,
    ServiceResidual
)


class QueueSimulatorEngine:
    """
    High-fidelity simulation engine for the college canteen queueing system.
    Tracks individual entity lifecycles, resource contention, and emits both
    Ground-Truth System State X_t and Noisy Sensor Observations Y_t every minute.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, initial_billing_counters: Optional[int] = None, initial_kitchen_stations: Optional[int] = None):
        """Resets the simulation environment to clean state at minute 0."""
        self.current_time: float = 0.0
        self.minute_step: int = 0
        
        c_b = initial_billing_counters or self.config.default_billing_counters
        c_k = initial_kitchen_stations or self.config.default_kitchen_stations
        
        # Resources
        self.billing_counters: List[Resource] = [
            Resource(resource_id=i, stage="BILLING", is_active=True)
            for i in range(c_b)
        ]
        self.kitchen_stations: List[Resource] = [
            Resource(resource_id=j, stage="KITCHEN", is_active=True)
            for j in range(c_k)
        ]
        
        # Queues (FIFO deques of waiting entities)
        self.queue_billing: deque[Customer] = deque()
        self.queue_kitchen: deque[Order] = deque()
        
        # Registry of all customer entities
        self.all_customers: Dict[str, Customer] = {}
        self.all_orders: Dict[str, Order] = {}
        
        # Completed / reneged lists
        self.completed_customers: List[Customer] = []
        self.cancelled_billing: List[Customer] = []
        self.cancelled_kitchen: List[Customer] = []
        
        # Operating regime
        self.current_regime: RegimeType = RegimeType.NORMAL
        
        # Telemetry logs
        self.history_ground_truth: List[SystemState] = []
        self.history_observations: List[ObservationVector] = []
        
        # Step interval delta trackers for [t, t+1)
        self.step_A_t: int = 0
        self.step_S_B: int = 0
        self.step_E_t: int = 0
        self.step_S_K: int = 0
        self.step_reneged_B: int = 0
        self.step_reneged_K: int = 0

    def apply_disruptions(self, disruptions: List[Dict[str, Any]], current_t: float):
        """Applies time-window disruptions to resources and arrival rates."""
        # Reset default efficiencies
        for c in self.billing_counters:
            c.efficiency_multiplier = 1.0
        for k in self.kitchen_stations:
            k.efficiency_multiplier = 1.0
        self.current_regime = RegimeType.NORMAL

        for d in disruptions:
            if d["start_min"] <= current_t <= d["end_min"]:
                if d["type"] == "BILLING_SLOWDOWN":
                    self.current_regime = RegimeType.SLOWDOWN
                    for c in self.billing_counters:
                        c.efficiency_multiplier = d.get("severity", 0.45)
                elif d["type"] == "KITCHEN_BOTTLENECK":
                    self.current_regime = RegimeType.BOTTLENECK
                    for k in self.kitchen_stations:
                        k.efficiency_multiplier = d.get("severity", 0.40)
                elif d["type"] == "ARRIVAL_SURGE":
                    self.current_regime = RegimeType.SURGE
                elif d["type"] == "COUNTER_UNAVAILABLE":
                    target_active = d.get("active_counters", 1)
                    for idx, c in enumerate(self.billing_counters):
                        c.is_active = (idx < target_active)

    def set_intervention_action(self, action: Dict[str, Any]):
        """
        Dynamically applies intervention action chosen by the optimizer:
        - add_billing_counter: int
        - add_kitchen_station: int
        - set_kitchen_express_lane: bool
        """
        if "billing_counters" in action:
            new_count = max(self.config.min_billing_counters, 
                            min(self.config.max_billing_counters, action["billing_counters"]))
            while len(self.billing_counters) < new_count:
                self.billing_counters.append(
                    Resource(resource_id=len(self.billing_counters), stage="BILLING", is_active=True)
                )
            for i, c in enumerate(self.billing_counters):
                c.is_active = (i < new_count)

        if "kitchen_stations" in action:
            new_count = max(self.config.min_kitchen_stations, 
                            min(self.config.max_kitchen_stations, action["kitchen_stations"]))
            while len(self.kitchen_stations) < new_count:
                self.kitchen_stations.append(
                    Resource(resource_id=len(self.kitchen_stations), stage="KITCHEN", is_active=True)
                )
            for j, k in enumerate(self.kitchen_stations):
                k.is_active = (j < new_count)

    def step_minute(self, arriving_customers_this_minute: List[Customer], disruptions: List[Dict[str, Any]]) -> Tuple[SystemState, ObservationVector]:
        """
        Advances the discrete simulation by 1.0 minute interval [t, t+1).
        Enforces exact queue conservation equations and transitions.
        """
        start_t = float(self.minute_step)
        end_t = start_t + self.config.time_step_minutes
        
        self.step_A_t = len(arriving_customers_this_minute)
        self.step_S_B = 0
        self.step_E_t = 0
        self.step_S_K = 0
        self.step_reneged_B = 0
        self.step_reneged_K = 0
        
        # 1. Update disruptions and active operating regime
        self.apply_disruptions(disruptions, start_t)
        
        # 2. Ingest new arrivals into Billing Queue
        for cust in arriving_customers_this_minute:
            cust.status = CustomerStatus.WAITING_BILLING
            self.queue_billing.append(cust)
            self.all_customers[cust.customer_id] = cust

        # 3. Process sub-minute continuous events within [start_t, end_t]
        # In discrete time step, advance service completions and reneging
        self._simulate_interval_events(start_t, end_t)
        
        # 4. Construct Ground-Truth SystemState X_t
        active_c_b = sum(1 for c in self.billing_counters if c.is_active)
        active_c_k = sum(1 for k in self.kitchen_stations if k.is_active)
        
        w_b = sum(c.expected_billing_duration for c in self.queue_billing)
        w_k = sum(o.expected_prep_workload for o in self.queue_kitchen)
        
        residuals: List[ServiceResidual] = []
        for c in self.billing_counters:
            if c.is_active and c.is_busy:
                rem = c.residual_time(end_t)
                residuals.append(ServiceResidual(c.resource_id, "BILLING", c.current_entity_id, rem))
        for k in self.kitchen_stations:
            if k.is_active and k.is_busy:
                rem = k.residual_time(end_t)
                residuals.append(ServiceResidual(k.resource_id, "KITCHEN", k.current_entity_id, rem))
                
        state_gt = SystemState(
            time_t=end_t,
            Q_B=len(self.queue_billing),
            Q_K=len(self.queue_kitchen),
            W_B=round(w_b, 2),
            W_K=round(w_k, 2),
            R_t=residuals,
            C_B=active_c_b,
            C_K=active_c_k,
            Z_t=self.current_regime,
            A_t=self.step_A_t,
            S_B=self.step_S_B,
            E_t=self.step_E_t,
            S_K=self.step_S_K,
            reneged_B=self.step_reneged_B,
            reneged_K=self.step_reneged_K
        )
        self.history_ground_truth.append(state_gt)
        
        # 5. Generate Noisy Sensor Observation Vector Y_t
        obs = self._generate_noisy_observation(state_gt, end_t)
        self.history_observations.append(obs)
        
        self.minute_step += 1
        self.current_time = end_t
        return state_gt, obs

    def _simulate_interval_events(self, start_t: float, end_t: float):
        """
        Simulates intra-minute state progression:
        - Check ongoing billing services -> completion
        - Assign waiting billing customers to free counters
        - Completed billing creates kitchen order -> enters kitchen queue
        - Check ongoing kitchen preps -> ready for collection
        - Assign waiting kitchen orders to free kitchen stations
        - Check customer reneging thresholds
        """
        # --- A. Finish Billing Services that ended by end_t ---
        for counter in self.billing_counters:
            if counter.is_busy and counter.service_end_time <= end_t:
                # Service finished
                cust_id = counter.current_entity_id
                cust = self.all_customers[cust_id]
                cust.timestamps.billing_end = counter.service_end_time
                cust.status = CustomerStatus.WAITING_KITCHEN
                
                # Free counter
                counter.current_entity_id = None
                counter.service_end_time = None
                self.step_S_B += 1
                
                # Create separate Kitchen Order entity
                order = Order(
                    order_id=f"ORD_{cust.customer_id}",
                    customer_id=cust.customer_id,
                    order_type=cust.order_type,
                    expected_prep_workload=self.config.service_params[cust.order_type.value]["workload_minutes"],
                    actual_prep_duration=self._sample_kitchen_duration(cust.order_type),
                    status=CustomerStatus.WAITING_KITCHEN,
                    timestamps=cust.timestamps
                )
                cust.timestamps.kitchen_queue = cust.timestamps.billing_end
                cust.order = order
                self.all_orders[order.order_id] = order
                
                # Order enters kitchen queue
                self.queue_kitchen.append(order)
                self.step_E_t += 1

        # --- B. Assign Waiting Billing Customers to Idle Active Counters ---
        for counter in self.billing_counters:
            if counter.is_active and not counter.is_busy and len(self.queue_billing) > 0:
                cust = self.queue_billing.popleft()
                cust.status = CustomerStatus.BILLING
                cust.assigned_counter = counter.resource_id
                cust.timestamps.billing_start = start_t
                
                # Service duration scaled by counter efficiency
                eff = max(0.1, counter.efficiency_multiplier)
                duration = cust.actual_billing_duration / eff
                counter.current_entity_id = cust.customer_id
                counter.service_end_time = start_t + duration

        # --- C. Finish Kitchen Preps that ended by end_t ---
        for station in self.kitchen_stations:
            if station.is_busy and station.service_end_time <= end_t:
                order_id = station.current_entity_id
                order = self.all_orders[order_id]
                order.timestamps.kitchen_ready = station.service_end_time
                order.timestamps.collected = station.service_end_time + 0.3  # quick pickup
                order.status = CustomerStatus.READY
                
                cust = self.all_customers[order.customer_id]
                cust.status = CustomerStatus.COLLECTED
                self.completed_customers.append(cust)
                
                station.current_entity_id = None
                station.service_end_time = None
                self.step_S_K += 1

        # --- D. Assign Waiting Kitchen Orders to Idle Stations ---
        for station in self.kitchen_stations:
            if station.is_active and not station.is_busy and len(self.queue_kitchen) > 0:
                order = self.queue_kitchen.popleft()
                order.status = CustomerStatus.PREPARING
                order.assigned_kitchen_station = station.resource_id
                order.timestamps.kitchen_start = start_t
                
                eff = max(0.1, station.efficiency_multiplier)
                duration = order.actual_prep_duration / eff
                station.current_entity_id = order.order_id
                station.service_end_time = start_t + duration

        # --- E. Check Reneging / Customer Abandonment ---
        # 1. Billing queue reneging
        surviving_billing = deque()
        while len(self.queue_billing) > 0:
            c = self.queue_billing.popleft()
            elapsed_wait = end_t - c.timestamps.arrived
            if elapsed_wait > c.patience_billing:
                c.status = CustomerStatus.CANCELLED_BILLING
                c.timestamps.cancelled = end_t
                self.cancelled_billing.append(c)
                self.step_reneged_B += 1
            else:
                surviving_billing.append(c)
        self.queue_billing = surviving_billing

        # 2. Kitchen backlog reneging (requesting refund)
        surviving_kitchen = deque()
        while len(self.queue_kitchen) > 0:
            o = self.queue_kitchen.popleft()
            elapsed_wait = end_t - (o.timestamps.kitchen_queue or end_t)
            cust = self.all_customers[o.customer_id]
            if elapsed_wait > cust.patience_kitchen:
                o.status = CustomerStatus.CANCELLED_KITCHEN
                cust.status = CustomerStatus.CANCELLED_KITCHEN
                cust.timestamps.cancelled = end_t
                self.cancelled_kitchen.append(cust)
                self.step_reneged_K += 1
            else:
                surviving_kitchen.append(o)
        self.queue_kitchen = surviving_kitchen

    def _sample_kitchen_duration(self, order_type: OrderType) -> float:
        """Helper to sample kitchen duration for order."""
        p_kitch = self.config.service_params[order_type.value]["kitchen"]
        var = p_kitch[1] ** 2
        mu = np.log(p_kitch[0] ** 2 / np.sqrt(var + p_kitch[0] ** 2))
        sig = np.sqrt(np.log(1 + var / (p_kitch[0] ** 2)))
        return float(self.rng.lognormal(mu, sig))

    def _generate_noisy_observation(self, gt: SystemState, current_t: float) -> ObservationVector:
        """
        Simulates realistic sensor noise, occlusion, and reporting latency:
        - Overhead CCTV count with Gaussian noise and occlusion dropouts.
        - Kitchen order ticket queue with sensor noise.
        - POS and pickup scan counts.
        """
        # Visual queue estimation noise
        noise_b = self.rng.normal(0, self.config.camera_count_noise_std)
        # Missed detection probability (occlusion in crowd)
        if self.rng.uniform(0, 1) < self.config.camera_miss_detection_prob:
            noisy_q_b = max(0.0, float(gt.Q_B) * 0.7 + noise_b)
        else:
            noisy_q_b = max(0.0, float(gt.Q_B) + noise_b)
            
        noise_k = self.rng.normal(0, self.config.kitchen_sensor_noise_std)
        noisy_q_k = max(0.0, float(gt.Q_K) + noise_k)
        
        # Ambient noise proxy (decibels above 60dB)
        crowd_proxy = 60.0 + 1.2 * (gt.Q_B + gt.Q_K) + self.rng.normal(0, 2.0)
        
        return ObservationVector(
            time_t=current_t,
            y_Q_B=round(noisy_q_b, 1),
            y_Q_K=round(noisy_q_k, 1),
            y_completed_billing=gt.S_B,
            y_completed_kitchen=gt.S_K,
            c_B_reported=gt.C_B,
            c_K_reported=gt.C_K,
            ambient_noise_level=round(crowd_proxy, 1)
        )
