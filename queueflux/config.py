"""
Configuration parameters and constants for QueueFlux.
Defines system topology, service distributions, operating regimes, and optimization parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SystemConfig:
    """Core configuration for the 2-stage canteen queueing system."""
    
    # Simulation Horizon
    minutes_per_day: int = 720          # 12 hours operating day (08:00 to 20:00)
    time_step_minutes: float = 1.0      # Discrete snapshot reporting interval
    default_days: int = 30              # Tonic AI synthetic data duration
    random_seed: int = 42
    
    # Stage A: Billing Configuration
    default_billing_counters: int = 3
    min_billing_counters: int = 1
    max_billing_counters: int = 5
    
    # Stage B: Kitchen Food Preparation Configuration
    default_kitchen_stations: int = 4
    min_kitchen_stations: int = 2
    max_kitchen_stations: int = 7
    
    # Service Duration Distributions by Order Type (LogNormal parameters: mean, std in minutes)
    # Quick: Beverages, packaged snacks, quick bakery items
    # Standard: Regular combo meal, thali, sandwich, rice bowl
    # Complex: Customized cook-to-order, special griddle/fryer combos
    service_params: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "QUICK": {
            "billing": (0.6, 0.15),      # mean 36s, std 9s
            "kitchen": (1.5, 0.40),      # mean 90s, std 24s
            "probability": 0.40,
            "workload_minutes": 1.5      # Expected kitchen workload unit
        },
        "STANDARD": {
            "billing": (1.0, 0.25),      # mean 60s, std 15s
            "kitchen": (4.0, 0.80),      # mean 240s, std 48s
            "probability": 0.45,
            "workload_minutes": 4.0
        },
        "COMPLEX": {
            "billing": (1.6, 0.35),      # mean 96s, std 21s
            "kitchen": (8.0, 1.80),      # mean 480s, std 108s
            "probability": 0.15,
            "workload_minutes": 8.0
        }
    })
    
    # Customer Reneging / Patience Thresholds (in minutes)
    # If wait in billing exceeds patience -> customer leaves without ordering
    # If wait in kitchen backlog exceeds patience -> customer cancels and asks for refund
    patience_billing_mean: float = 12.0
    patience_billing_std: float = 3.5
    patience_kitchen_mean: float = 20.0
    patience_kitchen_std: float = 5.0
    
    # Sensor Observation Noise (Simulating real-world noisy camera & IoT sensors)
    camera_count_noise_std: float = 1.2         # Gaussian noise on visual queue length
    camera_miss_detection_prob: float = 0.05    # Occlusion probability
    kitchen_sensor_noise_std: float = 1.0       # Ticket/shelf sensor noise
    
    # Operating Regimes
    regime_arrival_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL": 1.0,
        "SURGE": 2.2,          # Sudden rush between classes or rain
        "SLOWDOWN": 1.0,       # Regular demand but degraded throughput
        "BOTTLENECK": 1.1      # High demand hitting kitchen constraint
    })
    
    regime_service_multiplier_billing: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL": 1.0,
        "SURGE": 0.95,         # Slight degradation under stress
        "SLOWDOWN": 0.45,      # POS terminal glitch / network latency
        "BOTTLENECK": 0.90
    })
    
    regime_service_multiplier_kitchen: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL": 1.0,
        "SURGE": 0.90,
        "SLOWDOWN": 0.85,
        "BOTTLENECK": 0.40     # Appliance outage / fryer downtime
    })
    
    # Optimization / Intervention Costs
    cost_wait_minute_customer: float = 1.0     # Penalty per customer per minute of delay
    cost_cancellation_penalty: float = 50.0    # High penalty for customer balking / refund
    cost_counter_activation: float = 15.0      # Fixed operational cost to spin up extra counter
    cost_staff_shift_penalty: float = 8.0      # Friction cost for cross-training reallocation
    cost_express_batch_penalty: float = 10.0   # Prep deviation cost
    cvar_alpha: float = 0.95                   # 95% Conditional Value-at-Risk
    risk_aversion_lambda: float = 0.35         # Weight of tail risk in utility function


DEFAULT_CONFIG = SystemConfig()
