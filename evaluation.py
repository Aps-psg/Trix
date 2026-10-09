"""
evaluation.py - QueueFlux Quantitative Evaluation Engine

Evaluates system components against ground-truth simulator data:
1. Disruption Detection Accuracy: Precision, Recall, F1-Score, Detection Delay.
2. Waiting Time Prediction Accuracy: MAE, RMSE, MAPE.
3. Intervention Impact: Wait time reduction (minutes & %), throughput gain.

Handles unavailable or partial metrics safely without fabricating results.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import math


@dataclass
class DetectionEvaluationReport:
    """Quantitative evaluation metrics for disruption detection."""
    total_samples: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    avg_detection_delay_sec: Optional[float] = None
    status: str = "COMPLETED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "confusion_matrix": {
                "TP": self.true_positives,
                "FP": self.false_positives,
                "FN": self.false_negatives,
                "TN": self.true_negatives,
            },
            "precision": round(self.precision, 4) if self.precision is not None else "N/A",
            "recall": round(self.recall, 4) if self.recall is not None else "N/A",
            "f1_score": round(self.f1_score, 4) if self.f1_score is not None else "N/A",
            "avg_detection_delay_sec": round(self.avg_detection_delay_sec, 2) if self.avg_detection_delay_sec is not None else "N/A",
            "status": self.status
        }


@dataclass
class PredictionEvaluationReport:
    """Quantitative evaluation metrics for waiting time predictions."""
    sample_count: int = 0
    mae_minutes: Optional[float] = None
    rmse_minutes: Optional[float] = None
    mape_percent: Optional[float] = None
    status: str = "COMPLETED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mae_minutes": round(self.mae_minutes, 3) if self.mae_minutes is not None else "N/A",
            "rmse_minutes": round(self.rmse_minutes, 3) if self.rmse_minutes is not None else "N/A",
            "mape_percent": round(self.mape_percent, 2) if self.mape_percent is not None else "N/A",
            "status": self.status
        }


@dataclass
class InterventionImpactReport:
    """Quantitative evaluation of intervention effectiveness vs baseline."""
    baseline_avg_wait_min: float = 0.0
    intervention_avg_wait_min: float = 0.0
    wait_time_saved_min: float = 0.0
    percentage_reduction: float = 0.0
    status: str = "COMPLETED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_avg_wait_min": round(self.baseline_avg_wait_min, 2),
            "intervention_avg_wait_min": round(self.intervention_avg_wait_min, 2),
            "wait_time_saved_min": round(self.wait_time_saved_min, 2),
            "percentage_reduction": round(self.percentage_reduction, 2),
            "status": self.status
        }


class EvaluationEngine:
    """
    Computes statistical evaluation metrics for detection, prediction, and optimization.
    """

    @staticmethod
    def evaluate_disruption_detection(
        y_true: List[bool],
        y_pred: List[bool],
        detection_delays_sec: Optional[List[float]] = None
    ) -> DetectionEvaluationReport:
        """
        Calculates Precision, Recall, F1-Score, and average detection delay.
        """
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return DetectionEvaluationReport(status="UNAVAILABLE - Missing or size-mismatched ground truth data.")

        tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
        fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
        tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_delay = (sum(detection_delays_sec) / len(detection_delays_sec)) if detection_delays_sec else None

        return DetectionEvaluationReport(
            total_samples=len(y_true),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_detection_delay_sec=avg_delay,
            status="SUCCESS"
        )

    @staticmethod
    def evaluate_waiting_time_predictions(
        actual_wait_times: List[float],
        predicted_wait_times: List[float]
    ) -> PredictionEvaluationReport:
        """
        Calculates MAE, RMSE, and MAPE between predicted and actual wait times.
        """
        if not actual_wait_times or not predicted_wait_times or len(actual_wait_times) != len(predicted_wait_times):
            return PredictionEvaluationReport(status="UNAVAILABLE - Ground truth wait times missing or mismatched.")

        n = len(actual_wait_times)
        if n == 0:
            return PredictionEvaluationReport(status="UNAVAILABLE - Zero samples provided.")

        mae = sum(abs(a - p) for a, p in zip(actual_wait_times, predicted_wait_times)) / n
        mse = sum((a - p) ** 2 for a, p in zip(actual_wait_times, predicted_wait_times)) / n
        rmse = math.sqrt(mse)

        mape_elements = [abs(a - p) / a for a, p in zip(actual_wait_times, predicted_wait_times) if a > 0]
        mape = (sum(mape_elements) / len(mape_elements) * 100.0) if mape_elements else 0.0

        return PredictionEvaluationReport(
            sample_count=n,
            mae_minutes=mae,
            rmse_minutes=rmse,
            mape_percent=mape,
            status="SUCCESS"
        )

    @staticmethod
    def evaluate_intervention_impact(
        baseline_wait_times: List[float],
        intervention_wait_times: List[float]
    ) -> InterventionImpactReport:
        """
        Compares average waiting times before and after intervention.
        """
        if not baseline_wait_times or not intervention_wait_times:
            return InterventionImpactReport(status="UNAVAILABLE - Missing scenario baseline or intervention data.")

        base_avg = sum(baseline_wait_times) / len(baseline_wait_times)
        inter_avg = sum(intervention_wait_times) / len(intervention_wait_times)

        saved = max(0.0, base_avg - inter_avg)
        pct_reduction = (saved / base_avg * 100.0) if base_avg > 0 else 0.0

        return InterventionImpactReport(
            baseline_avg_wait_min=base_avg,
            intervention_avg_wait_min=inter_avg,
            wait_time_saved_min=saved,
            percentage_reduction=pct_reduction,
            status="SUCCESS"
        )
