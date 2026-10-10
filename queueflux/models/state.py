"""
Mathematical state definitions and observation models for QueueFlux.
Implements the 7-tuple state vector:
X_t = (Q_t^B, Q_t^K, W_t^B, W_t^K, R_t, C_t, Z_t)
Strictly separates Ground-Truth State from Inferred/Estimated State.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from queueflux.models.entities import RegimeType


@dataclass
class ServiceResidual:
    """Residual service time of an entity currently being served."""
    resource_id: int
    stage: str                          # "BILLING" or "KITCHEN"
    entity_id: str
    remaining_time: float               # Residual time in minutes


@dataclass
class SystemState:
    """
    Ground-Truth System State at time t.
    Accessible strictly by the simulator and evaluation pipeline.
    """
    time_t: float                       # Current simulation time (minutes)
    
    # Queue Lengths (waiting counts, excluding those currently at counter/station)
    Q_B: int                            # Number of customers waiting in billing queue
    Q_K: int                            # Number of orders waiting in kitchen queue
    
    # Workload (expected service-minutes of backlog)
    W_B: float                          # Total remaining billing workload (minutes)
    W_K: float                          # Total remaining kitchen workload (minutes)
    
    # In-service entities and residual times
    R_t: List[ServiceResidual] = field(default_factory=list)
    
    # Active resources: (c_B, c_K)
    C_B: int = 3                        # Active billing counters
    C_K: int = 4                        # Active kitchen stations
    
    # Operating regime
    Z_t: RegimeType = RegimeType.NORMAL
    
    # Accounting tallies for time step [t, t+1)
    A_t: int = 0                        # New arrivals in interval
    S_B: int = 0                        # Billing completions
    E_t: int = 0                        # Orders entering kitchen queue
    S_K: int = 0                        # Kitchen completions
    reneged_B: int = 0                  # Reneging during billing wait
    reneged_K: int = 0                  # Reneging during kitchen wait

    @property
    def total_residual_billing(self) -> float:
        return sum(r.remaining_time for r in self.R_t if r.stage == "BILLING")
        
    @property
    def total_residual_kitchen(self) -> float:
        return sum(r.remaining_time for r in self.R_t if r.stage == "KITCHEN")


@dataclass
class ObservationVector:
    """
    Noisy, incomplete sensor observations Y_t available to the QueueFlux observer.
    Generated from real-world telemetry (cameras, POS, kitchen ticket printers).
    """
    time_t: float
    y_Q_B: float                        # Noisy overhead camera customer count (with occlusion/drift)
    y_Q_K: float                        # Order display ticket count (lagged, missing offline orders)
    y_completed_billing: int            # POS transactions recorded in past minute
    y_completed_kitchen: int            # Barcode scanner pickups in past minute
    c_B_reported: int                   # Active counters reported by POS login
    c_K_reported: int                   # Kitchen staff clocked in
    ambient_noise_level: float = 0.0    # Canteen hall decibel proxy for rush/crowding


@dataclass
class EstimatedState:
    """
    Inferred/Estimated State produced by QueueFlux Observer under uncertainty.
    Includes point estimates and credible intervals (P10, P50, P90).
    """
    time_t: float
    
    # Estimated Queue Lengths
    Q_B_hat: float
    Q_B_p10: float
    Q_B_p90: float
    
    Q_K_hat: float
    Q_K_p10: float
    Q_K_p90: float
    
    # Estimated Workload (in service-minutes)
    W_B_hat: float
    W_B_p10: float
    W_B_p90: float
    
    W_K_hat: float
    W_K_p10: float
    W_K_p90: float
    
    # Estimated Residual in-service load
    residual_B_hat: float
    residual_K_hat: float
    
    # Estimated Active Capacity
    C_B_hat: float
    C_K_hat: float
    
    # Inferred Operating Regime Probabilities: P(Z_t = z)
    regime_probs: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL": 0.8,
        "SURGE": 0.1,
        "SLOWDOWN": 0.05,
        "BOTTLENECK": 0.05
    })
    
    @property
    def most_likely_regime(self) -> RegimeType:
        top = max(self.regime_probs.items(), key=lambda kv: kv[1])[0]
        return RegimeType(top)
