from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .metrics import compute_binary_metrics, curve_payload, threshold_sweep


# =========================
# DATA STRUCTURE
# =========================
@dataclass
class EvaluationResult:
    name: str
    threshold: float
    metrics: dict
    curves: dict
    probabilities: np.ndarray
    predictions: np.ndarray


# =========================
# MAIN EVALUATOR
# =========================
class ResearchEvaluator:

    def __init__(
        self,
        recall_floor: float = 0.90,
        far_ceiling: float | None = None,
        grid_size: int = 101
    ):
        self.recall_floor = recall_floor
        self.far_ceiling = far_ceiling
        self.grid_size = grid_size

    # =========================
    # THRESHOLD SELECTION (FIXED)
    # =========================
    def choose_threshold(self, name, y_true, y_prob):
        # Unified grid for all models — fairness and consistency.
        thresholds = np.linspace(0.01, 0.9, self.grid_size)

        sweep = []
        for t in thresholds:
            m = compute_binary_metrics(y_true, y_prob, float(t))
            m['meets_recall_floor'] = m['recall'] >= self.recall_floor
            m['meets_far_ceiling'] = (
                True if self.far_ceiling is None
                else m['false_alarm_rate'] <= self.far_ceiling
            )
            sweep.append(m)

        sweep = []

        for t in thresholds:
            m = compute_binary_metrics(y_true, y_prob, float(t))

            # constraints
            m['meets_recall_floor'] = m['recall'] >= self.recall_floor
            m['meets_far_ceiling'] = (
                True if self.far_ceiling is None
                else m['false_alarm_rate'] <= self.far_ceiling
            )

            sweep.append(m)

        # =========================
        # FILTER FEASIBLE REGION
        # =========================
        feasible = [
            m for m in sweep
            if m['meets_recall_floor'] and m['meets_far_ceiling']
        ]

        pool = feasible if feasible else sweep

        # =========================
        # 🔥 FIXED OBJECTIVE (CRITICAL)
        # =========================
        pool = sorted(
            pool,
            key=lambda m: m['f1'],
            reverse=True
        )

        best = pool[0]

        return float(best['threshold']), sweep

    # =========================
    # FINAL EVALUATION (FIXED)
    # =========================
    def evaluate(self, name, y_true, y_prob, fixed_threshold=None):

        # =========================
        # THRESHOLD SELECTION
        # =========================
        if fixed_threshold is not None:
            t = float(fixed_threshold)

            # still compute sweep for analysis
            sweep = threshold_sweep(
                y_true,
                y_prob,
                self.grid_size,
                self.recall_floor,
                self.far_ceiling,
                model_name=name
            )

        else:
            t, sweep = self.choose_threshold(name, y_true, y_prob)

            # 🔥 safer lower bound (prevents degenerate threshold)
            t = max(t, 0.01)

        # =========================
        # METRICS
        # =========================
        metrics = compute_binary_metrics(y_true, y_prob, t)
        curves = curve_payload(y_true, y_prob)
        pred = (y_prob >= t).astype(int)

        return EvaluationResult(
            name=name,
            threshold=float(t),
            metrics=metrics,
            curves=curves,
            probabilities=y_prob.astype(np.float32),
            predictions=pred.astype(np.int32)
        ), sweep

    # =========================
    # TABLE FORMAT
    # =========================
    @staticmethod
    def to_table(results):
        rows = []

        for r in results:
            rows.append({
                'Model': r.name,
                'Threshold': round(r.threshold, 6),
                'AUROC': round(r.metrics['auroc'], 6),
                'AUPRC': round(r.metrics['auprc'], 6),
                'F1': round(r.metrics['f1'], 6),
                'DetectionRate': round(r.metrics['detection_rate'], 6),
                'FalseAlarmRate': round(r.metrics['false_alarm_rate'], 6),
                'Precision': round(r.metrics['precision'], 6),
                'Recall': round(r.metrics['recall'], 6),
            })

        return pd.DataFrame(rows)