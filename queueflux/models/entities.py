"""
Entity definitions for students, food orders, and operational resources in QueueFlux.
Strictly models lifecycle: ARRIVED -> WAITING_BILLING -> BILLING -> WAITING_KITCHEN -> PREPARING -> READY -> COLLECTED
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class CustomerStatus(str, Enum):
    ARRIVED = "ARRIVED"
    WAITING_BILLING = "WAITING_BILLING"
    BILLING = "BILLING"
    WAITING_KITCHEN = "WAITING_KITCHEN"
    PREPARING = "PREPARING"
    READY = "READY"
    COLLECTED = "COLLECTED"
    CANCELLED_BILLING = "CANCELLED_BILLING"
    CANCELLED_KITCHEN = "CANCELLED_KITCHEN"


class OrderType(str, Enum):
    QUICK = "QUICK"
    STANDARD = "STANDARD"
    COMPLEX = "COMPLEX"


class RegimeType(str, Enum):
    NORMAL = "NORMAL"
    SURGE = "SURGE"
    SLOWDOWN = "SLOWDOWN"
    BOTTLENECK = "BOTTLENECK"


@dataclass
class LifecycleTimestamps:
    """Exact simulation timestamps (in minutes from day start) for evaluation."""
    arrived: float
    billing_start: Optional[float] = None
    billing_end: Optional[float] = None
    kitchen_queue: Optional[float] = None
    kitchen_start: Optional[float] = None
    kitchen_ready: Optional[float] = None
    collected: Optional[float] = None
    cancelled: Optional[float] = None


@dataclass
class Order:
    """
    Food preparation order entity.
    Maintained independently from customer billing entity.
    """
    order_id: str
    customer_id: str
    order_type: OrderType
    expected_prep_workload: float          # Workload units (nominal minutes)
    actual_prep_duration: float            # Sampled realization (minutes)
    status: CustomerStatus = CustomerStatus.WAITING_KITCHEN
    timestamps: Optional[LifecycleTimestamps] = None
    assigned_kitchen_station: Optional[int] = None
    
    @property
    def is_completed(self) -> bool:
        return self.status in (CustomerStatus.READY, CustomerStatus.COLLECTED)
    
    @property
    def is_active_in_kitchen(self) -> bool:
        return self.status in (CustomerStatus.WAITING_KITCHEN, CustomerStatus.PREPARING)


@dataclass
class Customer:
    """
    Customer entity arriving at the canteen.
    Carries order requirements, patience threshold, and tracking timestamps.
    """
    customer_id: str
    arrival_time: float                     # Timestamp in minutes
    order_type: OrderType
    expected_billing_duration: float        # Expected service duration
    actual_billing_duration: float          # Sampled realization
    patience_billing: float                 # Reneging threshold in billing queue (minutes)
    patience_kitchen: float                 # Reneging threshold in kitchen backlog (minutes)
    order: Optional[Order] = None           # Linked order created after billing completion
    status: CustomerStatus = CustomerStatus.ARRIVED
    assigned_counter: Optional[int] = None
    timestamps: LifecycleTimestamps = field(init=False)
    
    def __post_init__(self):
        self.timestamps = LifecycleTimestamps(arrived=self.arrival_time)
        
    @property
    def total_waiting_time(self) -> Optional[float]:
        """Calculates total customer delay: billing wait + kitchen wait."""
        b_wait = 0.0
        if self.timestamps.billing_start is not None:
            b_wait = self.timestamps.billing_start - self.timestamps.arrived
        elif self.timestamps.cancelled is not None and self.status == CustomerStatus.CANCELLED_BILLING:
            return self.timestamps.cancelled - self.timestamps.arrived
            
        k_wait = 0.0
        if self.timestamps.kitchen_ready is not None and self.timestamps.kitchen_queue is not None:
            k_wait = self.timestamps.kitchen_ready - self.timestamps.kitchen_queue
        elif self.timestamps.cancelled is not None and self.status == CustomerStatus.CANCELLED_KITCHEN and self.timestamps.kitchen_queue is not None:
            k_wait = self.timestamps.cancelled - self.timestamps.kitchen_queue
            
        return b_wait + k_wait


@dataclass
class Resource:
    """
    Operational resource: Billing Counter or Kitchen Preparation Station.
    Tracks residual service time and current serving entity.
    """
    resource_id: int
    stage: str                              # "BILLING" or "KITCHEN"
    is_active: bool = True
    efficiency_multiplier: float = 1.0      # Degraded when slowdown or disruption occurs
    current_entity_id: Optional[str] = None # Customer ID (billing) or Order ID (kitchen)
    service_end_time: Optional[float] = None
    
    def residual_time(self, current_time: float) -> float:
        """Returns remaining service duration in minutes."""
        if self.service_end_time is None or self.service_end_time <= current_time:
            return 0.0
        return max(0.0, self.service_end_time - current_time)
    
    @property
    def is_busy(self) -> bool:
        return self.current_entity_id is not None
