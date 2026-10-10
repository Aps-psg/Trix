"""QueueFlux — Person 1: ML Model & Queue Prediction
===================================================
Technical Responsibility: Core AI/ML Model Engineering
- Generates synthetic historical queue data (Tonic.ai schema).
- Preprocesses data and engineers queue lag/capacity features.
- Implements Baseline Model (Naive Moving Average / FIFO) vs. Improved ML Model (Multivariate Ridge Regression).
- Predicts current queue length, waiting time, and 5-minute ahead forecasts.
- Evaluates prediction accuracy via MAE and RMSE against the baseline.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


# ============================================================================
# 1. SYNTHETIC HISTORICAL DATA GENERATOR (TONIC.AI SCHEMA)
# ============================================================================
class TonicSyntheticQueueGenerator:
    """Generates synthetic historical queue records adhering to the agreed shared interface:

    [timestamp, arrival_rate, service_rate, queue_length, active_counters, avg_service_time]
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_historical_dataset(self, days: int = 14, minutes_per_day: int = 720) -> pd.DataFrame:
        """Generates multi-day operational queue telemetry spanning 12 hours (720 min) per day."""
        records: List[Dict[str, Any]] = []
        base_time = datetime.datetime.now() - datetime.timedelta(days=days)

        for day in range(days):
            current_date = base_time + datetime.timedelta(days=day)
            q_b = 2

            for minute in range(minutes_per_day):
                timestamp = current_date + datetime.timedelta(minutes=minute)

                # Time-of-day arrival rate pattern (peaks at lunch rush around minute 300-420)
                rush_effect = 2.8 * np.exp(-((minute - 360) ** 2) / (2 * (60 ** 2)))
                base_arr = 1.2 + rush_effect
                arr_rate = float(np.clip(self.rng.normal(base_arr, 0.4), 0.2, 8.0))

                # Active counters and service times
                active_counters = 3 if minute >= 300 and minute <= 450 else 2
                avg_svc_time = float(np.clip(self.rng.normal(1.2, 0.2), 0.6, 2.5))  # in minutes
                svc_rate = float(active_counters / avg_svc_time)

                # Queue evolution
                arrivals = self.rng.poisson(arr_rate)
                served = min(q_b, self.rng.poisson(svc_rate))
                q_b = max(0, q_b + arrivals - served)

                # Wait time calculation (Little's Law + renewal residual)
                wait_time = float((q_b * avg_svc_time / active_counters) + (avg_svc_time / 2.0))

                records.append({
                    "timestamp": timestamp.isoformat(),
                    "minute_of_day": minute,
                    "arrival_rate": round(arr_rate, 2),
                    "service_rate": round(svc_rate, 2),
                    "queue_length": int(q_b),
                    "active_counters": int(active_counters),
                    "avg_service_time": round(avg_svc_time, 2),
                    "actual_wait_time": round(wait_time, 2),
                })

        return pd.DataFrame(records)


# ============================================================================
# 2. FEATURE ENGINEERING & PREPROCESSING
# ============================================================================
def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Engineers queue utilization ratio, capacity gap, and polynomial terms."""
    arr = df["arrival_rate"].to_numpy(dtype=float)
    svc = df["service_rate"].to_numpy(dtype=float)
    cnt = df["active_counters"].to_numpy(dtype=float)
    ast = df["avg_service_time"].to_numpy(dtype=float)
    ql = df["queue_length"].to_numpy(dtype=float)

    # Features: [queue_length, arrival_rate, service_rate, active_counters, avg_service_time, utilization, capacity_deficit]
    utilization = arr / np.maximum(0.1, svc)
    capacity_deficit = np.maximum(0.0, arr - svc)
    workload = ql * ast / np.maximum(1.0, cnt)

    X = np.column_stack([
        np.ones_like(ql),
        ql,
        arr,
        svc,
        cnt,
        ast,
        utilization,
        capacity_deficit,
        workload,
    ])

    y_q = ql
    y_w = df["actual_wait_time"].to_numpy(dtype=float)
    return X, y_q, y_w


# ============================================================================
# 3. BASELINE VS. IMPROVED ML PREDICTION MODEL
# ============================================================================
class QueueMLPredictor:
    """Person 1 Machine Learning Queue & Waiting-Time Prediction Model.

    Compares Baseline (Naive Moving Average / FIFO) against Improved (Trained Ridge Regression).
    """

    def __init__(self) -> None:
        self.weights_q: np.ndarray | None = None
        self.weights_w: np.ndarray | None = None
        self.baseline_mae: float = 0.0
        self.baseline_rmse: float = 0.0
        self.improved_mae: float = 0.0
        self.improved_rmse: float = 0.0
        self.is_trained: bool = False
        self._train_default_model()

    def _train_default_model(self) -> None:
        """Trains the model on historical synthetic data generated using Tonic.ai schema."""
        gen = TonicSyntheticQueueGenerator(seed=101)
        df = gen.generate_historical_dataset(days=7, minutes_per_day=360)
        self.fit(df)

    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """Trains the baseline and improved prediction models on historical data."""
        X, y_q, y_w = extract_features(df)
        n = len(X)
        split = int(0.8 * n)

        X_train, X_test = X[:split], X[split:]
        yq_train, yq_test = y_q[:split], y_q[split:]
        yw_train, yw_test = y_w[:split], y_w[split:]

        # 1. Baseline Model: Naive moving average
        baseline_pred_q = np.roll(yq_test, 1)
        baseline_pred_q[0] = np.mean(yq_train)

        # Baseline error
        err_base = yq_test - baseline_pred_q
        self.baseline_mae = float(np.mean(np.abs(err_base)))
        self.baseline_rmse = float(np.sqrt(np.mean(err_base ** 2)))

        # 2. Improved Model: Regularized Multivariate Ridge Regression
        reg = 1.0
        xtx = X_train.T @ X_train + reg * np.eye(X_train.shape[1])
        self.weights_q = np.linalg.solve(xtx, X_train.T @ yq_train)
        self.weights_w = np.linalg.solve(xtx, X_train.T @ yw_train)

        # Evaluate Improved Model
        pred_q_test = np.maximum(0.0, X_test @ self.weights_q)
        err_imp = yq_test - pred_q_test
        self.improved_mae = float(np.mean(np.abs(err_imp)))
        self.improved_rmse = float(np.sqrt(np.mean(err_imp ** 2)))
        self.is_trained = True

        return {
            "baseline_mae": round(self.baseline_mae, 3),
            "baseline_rmse": round(self.baseline_rmse, 3),
            "improved_mae": round(self.improved_mae, 3),
            "improved_rmse": round(self.improved_rmse, 3),
            "mae_reduction_pct": round(
                ((self.baseline_mae - self.improved_mae) / max(0.01, self.baseline_mae)) * 100, 1
            ),
        }

    def predict(
        self,
        queue_length: int,
        arrival_rate: float,
        service_rate: float,
        active_counters: int,
        avg_service_time: float,
    ) -> Dict[str, Any]:
        """Predicts current queue length, waiting time, and 5-minute ahead forecasts."""
        cnt = max(1, active_counters)
        ast = max(0.1, avg_service_time)
        util = arrival_rate / max(0.1, service_rate)
        deficit = max(0.0, arrival_rate - service_rate)
        workload = queue_length * ast / cnt

        feat = np.array([
            1.0,
            float(queue_length),
            float(arrival_rate),
            float(service_rate),
            float(cnt),
            float(ast),
            float(util),
            float(deficit),
            float(workload),
        ])

        # Prediction for t
        pred_q = float(max(0.0, feat @ self.weights_q))
        pred_w = float(max(0.5, feat @ self.weights_w))

        # 5-Minute Ahead Forward Simulation / Projection (t+1, t+2, t+3, t+4, t+5)
        forecast_5min: List[float] = []
        forecast_waits: List[float] = []
        curr_q = pred_q
        for _ in range(5):
            curr_q = max(0.0, curr_q + arrival_rate - service_rate)
            curr_w = (curr_q * ast / cnt) + (ast / 2.0)
            forecast_5min.append(round(curr_q, 1))
            forecast_waits.append(round(curr_w, 1))

        return {
            "predicted_queue_length": round(pred_q, 1),
            "predicted_wait_time": round(pred_w, 1),
            "forecast_queue_5min": forecast_5min,
            "forecast_wait_5min": forecast_waits,
            "model_accuracy": {
                "baseline_mae": round(self.baseline_mae, 2),
                "baseline_rmse": round(self.baseline_rmse, 2),
                "model_mae": round(self.improved_mae, 2),
                "model_rmse": round(self.improved_rmse, 2),
            },
        }


# Global singleton instance
person1_predictor = QueueMLPredictor()
