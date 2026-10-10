"""
Prediction module for workload-aware waiting times and 5-minute probabilistic forecasts.
"""

from queueflux.predictor.waiting_time import WorkloadWaitingTimePredictor
from queueflux.predictor.forecaster import FiveMinuteQueueForecaster

__all__ = ["WorkloadWaitingTimePredictor", "FiveMinuteQueueForecaster"]
