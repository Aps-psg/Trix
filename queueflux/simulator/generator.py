"""
Tonic AI-inspired 30-day synthetic dataset generator for college canteen operations.
Generates reproducible, statistically realistic customer streams, order profiles,
time-varying arrival intensity, and operational anomalies.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from queueflux.config import SystemConfig, DEFAULT_CONFIG
from queueflux.models.entities import OrderType, RegimeType


class TonicSyntheticDataGenerator:
    """
    Generates synthetic canteen operational data for 30 consecutive operating days.
    Emulates Tonic AI synthetic data patterns:
    - Multi-modal non-homogeneous Poisson arrival processes (NHPP).
    - Class-schedule boundary pulses (surges at :50 to :10 of each hour).
    - Meal-specific order type correlations.
    - Realistic customer patience thresholds.
    - Injected real-world disruptions (hardware faults, supply bottlenecks, weather rushes).
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG, seed: int = 42):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _arrival_intensity_profile(self, minute_of_day: float, day_of_week: int) -> float:
        """
        Calculates expected arrival intensity lambda(t) in customers per minute.
        Day spans 0 to 720 minutes (8:00 AM to 8:00 PM).
        0 = 8:00 AM, 60 = 9:00 AM, 240 = 12:00 PM, 300 = 1:00 PM, 510 = 4:30 PM, 720 = 8:00 PM.
        """
        # Base off-peak rate
        base_rate = 0.8
        
        # Morning breakfast rush: 8:30 - 9:45 (t = 30 to 105)
        breakfast = 2.8 * np.exp(-((minute_of_day - 65) ** 2) / (2 * 25 ** 2))
        
        # Heavy Lunch rush: 12:00 - 14:00 (t = 240 to 360)
        lunch = 5.6 * np.exp(-((minute_of_day - 290) ** 2) / (2 * 45 ** 2))
        
        # Evening snack rush: 16:30 - 17:45 (t = 510 to 585)
        evening = 3.2 * np.exp(-((minute_of_day - 540) ** 2) / (2 * 30 ** 2))
        
        # Periodic college class transitions (lectures end around minute 50 of each hour)
        hour_fraction = minute_of_day % 60
        class_bell_pulse = 1.2 * np.exp(-((hour_fraction - 52) ** 2) / (2 * 4 ** 2))
        
        total_rate = base_rate + breakfast + lunch + evening + class_bell_pulse
        
        # Weekend adjustment (Day 5, 6: Sat, Sun) -> lower volume, flatter peaks
        if day_of_week in (5, 6):
            total_rate *= 0.55
            
        return max(0.2, total_rate)

    def _get_disruption_schedule(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Explicit 30-day disruption calendar to benchmark propagation & adaptation.
        Returns day_index -> list of disruption events.
        """
        return {
            3: [
                # Day 4 (index 3): Billing POS terminal outage during lunch rush
                {"type": "BILLING_SLOWDOWN", "start_min": 270, "end_min": 315, "severity": 0.40,
                 "description": "Network latency & POS gateway timeouts"}
            ],
            7: [
                # Day 8: Sudden cloudburst / torrential rain causing massive indoor surge
                {"type": "ARRIVAL_SURGE", "start_min": 250, "end_min": 320, "multiplier": 2.3,
                 "description": "Rain deluge forcing campus crowd into canteen"}
            ],
            13: [
                # Day 14: Main kitchen deep-fryer & grill tripping circuit breaker
                {"type": "KITCHEN_BOTTLENECK", "start_min": 280, "end_min": 340, "severity": 0.35,
                 "description": "Appliance outage reducing cooking capacity"}
            ],
            20: [
                # Day 21: Mid-term exam finish simultaneous release
                {"type": "ARRIVAL_SURGE", "start_min": 180, "end_min": 220, "multiplier": 2.5,
                 "description": "Exam dismissal surge across 4 lecture halls"}
            ],
            26: [
                # Day 27: Cashier shortage (only 1 counter active)
                {"type": "COUNTER_UNAVAILABLE", "start_min": 240, "end_min": 330, "active_counters": 1,
                 "description": "Staff emergency absence"}
            ]
        }

    def generate_day(self, day_index: int) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Generates synthetic customer arrivals and order characteristics for one day.
        Returns:
            df: DataFrame of arrived customers and orders
            events: list of scheduled operational events
        """
        day_of_week = day_index % 7  # 0=Monday, ..., 6=Sunday
        disruptions = self._get_disruption_schedule().get(day_index, [])
        
        customers = []
        cust_id_counter = 1
        
        # Simulate arrivals across the 720 minutes using thinning of Poisson process
        # Max theoretical arrival rate lambda_max across day
        lambda_max = 14.0
        
        current_t = 0.0
        while current_t < self.config.minutes_per_day:
            # Generate next candidate arrival time
            inter_arrival = self.rng.exponential(1.0 / lambda_max)
            current_t += inter_arrival
            if current_t >= self.config.minutes_per_day:
                break
                
            # Base arrival intensity
            rate_t = self._arrival_intensity_profile(current_t, day_of_week)
            
            # Apply surge disruption if active
            for d in disruptions:
                if d["type"] == "ARRIVAL_SURGE" and d["start_min"] <= current_t <= d["end_min"]:
                    rate_t *= d["multiplier"]
                    
            # Thinning acceptance
            if self.rng.uniform(0, 1) <= (rate_t / lambda_max):
                # Sample order type with time-dependent bias
                # Lunch time (210 to 360) has more STANDARD and COMPLEX meals
                if 210 <= current_t <= 360:
                    type_probs = [0.20, 0.55, 0.25]
                elif 30 <= current_t <= 120:  # Breakfast
                    type_probs = [0.65, 0.30, 0.05]
                else:  # Evening / Afternoon
                    type_probs = [0.45, 0.40, 0.15]
                    
                order_str = str(self.rng.choice(
                    ["QUICK", "STANDARD", "COMPLEX"],
                    p=type_probs
                ))
                order_choice = OrderType(order_str)
                
                # Sample billing and kitchen nominal durations from Lognormal
                p_bill = self.config.service_params[order_choice.value]["billing"]
                p_kitch = self.config.service_params[order_choice.value]["kitchen"]
                
                # Compute lognormal mu & sigma from given mean and std
                def calc_lognorm_params(mean, std):
                    var = std ** 2
                    mu = np.log(mean ** 2 / np.sqrt(var + mean ** 2))
                    sigma = np.sqrt(np.log(1 + var / (mean ** 2)))
                    return mu, sigma
                
                mu_b, sig_b = calc_lognorm_params(p_bill[0], p_bill[1])
                mu_k, sig_k = calc_lognorm_params(p_kitch[0], p_kitch[1])
                
                billing_dur = float(self.rng.lognormal(mu_b, sig_b))
                kitchen_dur = float(self.rng.lognormal(mu_k, sig_k))
                
                # Patience / Reneging thresholds (minutes)
                patience_b = float(self.rng.normal(self.config.patience_billing_mean, self.config.patience_billing_std))
                patience_b = max(4.0, patience_b)
                
                patience_k = float(self.rng.normal(self.config.patience_kitchen_mean, self.config.patience_kitchen_std))
                patience_k = max(8.0, patience_k)
                
                workload_unit = self.config.service_params[order_choice.value]["workload_minutes"]
                
                cid = f"D{day_index + 1:02d}_C{cust_id_counter:04d}"
                oid = f"D{day_index + 1:02d}_ORD{cust_id_counter:04d}"
                cust_id_counter += 1
                
                customers.append({
                    "day_index": day_index + 1,
                    "customer_id": cid,
                    "order_id": oid,
                    "arrival_minute": round(current_t, 3),
                    "order_type": order_choice.value,
                    "expected_billing_duration": round(p_bill[0], 2),
                    "sampled_billing_duration": round(billing_dur, 3),
                    "expected_prep_workload": round(workload_unit, 2),
                    "sampled_kitchen_duration": round(kitchen_dur, 3),
                    "patience_billing": round(patience_b, 2),
                    "patience_kitchen": round(patience_k, 2)
                })
                
        df = pd.DataFrame(customers)
        return df, disruptions

    def generate_full_30_days(self, output_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generates full 30 days of data and saves enterprise synthetic artifacts.
        """
        os.makedirs(output_dir, exist_ok=True)
        all_days = []
        event_log = {}
        
        print(f"[QueueFlux Synthetic Data Generator] Generating 30 days of realistic canteen traffic...")
        
        for d in range(self.config.default_days):
            df_day, events = self.generate_day(d)
            all_days.append(df_day)
            event_log[f"Day_{d + 1}"] = events
            
        full_df = pd.concat(all_days, ignore_index=True)
        
        # Save Tonic AI compatible outputs
        csv_path = os.path.join(output_dir, "synthetic_canteen_30d.csv")
        json_path = os.path.join(output_dir, "disruption_events_30d.json")
        summary_path = os.path.join(output_dir, "tonic_metadata_30d.json")
        
        full_df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(event_log, f, indent=2)
            
        metadata = {
            "dataset_name": "Tonic_QueueFlux_Canteen_30D",
            "total_days": self.config.default_days,
            "total_records": len(full_df),
            "order_type_distribution": full_df["order_type"].value_counts(normalize=True).to_dict(),
            "avg_arrivals_per_day": round(len(full_df) / self.config.default_days, 1),
            "disruption_days_count": len([k for k, v in event_log.items() if len(v) > 0])
        }
        with open(summary_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"[QueueFlux Synthetic Data Generator] Successfully generated {len(full_df):,} orders across 30 days.")
        print(f"  Artifacts saved to: {output_dir}")
        return full_df, metadata
