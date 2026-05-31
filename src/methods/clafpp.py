from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score


@dataclass
class RobustStats:
    center: float
    scale: float


@dataclass
class FusionArtifacts:
    component_stats: Dict[str, RobustStats]
    learned_weights: Dict[str, float]
    utility_scores: Dict[str, float]


class RobustNormalizer:
    """Median-IQR normalization followed by logistic squashing."""

    @staticmethod
    def fit(x: np.ndarray) -> RobustStats:
        x = np.asarray(x, dtype=np.float32)
        center = float(np.median(x))
        q75, q25 = np.percentile(x, [75, 25])
        scale = float(max(q75 - q25, 1e-6))
        return RobustStats(center=center, scale=scale)

    @staticmethod
    def transform(x: np.ndarray, stats: RobustStats) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        z = (x - stats.center) / stats.scale

        # safer normalization (no sigmoid compression)
        z_min = z.min()
        z_max = z.max()

        return ((z - z_min) / (z_max - z_min + 1e-6)).astype(np.float32)


class LearnedWeightedFusion:
    """Utility-weighted fusion used for the generative branch and soft voting."""

    def __init__(self, weight_floor: float = 0.05):
        self.weight_floor = float(weight_floor)
        self.artifacts_: FusionArtifacts | None = None

    def fit(self, y_ref: np.ndarray, component_dict: Dict[str, np.ndarray]) -> "LearnedWeightedFusion":
        stats = {k: RobustNormalizer.fit(v) for k, v in component_dict.items()}
        normalized = {k: RobustNormalizer.transform(v, stats[k]) for k, v in component_dict.items()}

        utilities = {}
        for name, values in normalized.items():
            if len(np.unique(y_ref)) > 1:
                utilities[name] = max(float(average_precision_score(y_ref, values)), self.weight_floor)
            else:
                utilities[name] = self.weight_floor

        total = sum(utilities.values())
        learned_weights = {k: float(v / total) for k, v in utilities.items()}
        self.artifacts_ = FusionArtifacts(component_stats=stats, learned_weights=learned_weights, utility_scores=utilities)
        return self

    def transform(self, component_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.artifacts_ is None:
            raise RuntimeError("Fusion must be fit before transform().")
        return {
            k: RobustNormalizer.transform(v, self.artifacts_.component_stats[k])
            for k, v in component_dict.items()
        }

    def score(self, component_dict: Dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
        normalized = self.transform(component_dict)
        fused = np.zeros(len(next(iter(normalized.values()))), dtype=np.float32)
        for name, values in normalized.items():
            fused += float(self.artifacts_.learned_weights[name]) * values.astype(np.float32)
        details = {
            "normalized_components": {k: v.astype(np.float32) for k, v in normalized.items()},
            "weights": self.artifacts_.learned_weights,
            "utilities": self.artifacts_.utility_scores,
        }
        return fused.astype(np.float32), details

    @property
    def weights_(self) -> Dict[str, float]:
        return {} if self.artifacts_ is None else self.artifacts_.learned_weights

    def state_dict(self) -> dict:
        if self.artifacts_ is None:
            return {}
        return {
            "component_stats": {
                k: {"center": float(v.center), "scale": float(v.scale)}
                for k, v in self.artifacts_.component_stats.items()
            },
            "learned_weights": self.artifacts_.learned_weights,
            "utility_scores": self.artifacts_.utility_scores,
        }


class SoftVotingFusion(LearnedWeightedFusion):
    """Baseline only. Same estimator, different research interpretation."""


class ResidualSymbolicEnhancer:
    """Calibrates a symbolic residual on top of the neural meta score.

    Final score:
        S(x) = clip(s_meta(x) + lambda * r_tilde(x), 0, 1)
    where r_tilde(x) is the symbolic score after the SAME z-score + sigmoid
    normalization used at fit time, and lambda is learned on validation data.
    """

    def __init__(self, lambda_grid: np.ndarray | None = None, objective: str = "auprc"):
        self.lambda_grid = lambda_grid if lambda_grid is not None else np.linspace(0.0, 1.0, 41)
        self.objective = objective
        self.lambda_: float = 0.0
        # Store normalization params so inference matches training.
        self.sym_mean_: float = 0.0
        self.sym_std_: float = 1.0

    def _normalize_symbolic(self, symbolic_score: np.ndarray) -> np.ndarray:
        z = (symbolic_score - self.sym_mean_) / (self.sym_std_ + 1e-6)
        return 1.0 / (1.0 + np.exp(-z))

    def _metric(self, y_true: np.ndarray, score: np.ndarray) -> float:
        if len(np.unique(y_true)) <= 1:
            return 0.0
        if self.objective == "auprc":
            return float(average_precision_score(y_true, score))
        pos = score[y_true == 1].mean() if np.any(y_true == 1) else 0.0
        neg = score[y_true == 0].mean() if np.any(y_true == 0) else 0.0
        return float(pos - neg)

    def fit(self, y_val: np.ndarray, neural_score: np.ndarray, symbolic_score: np.ndarray) -> "ResidualSymbolicEnhancer":
        # Fit normalization params on validation symbolic scores.
        self.sym_mean_ = float(symbolic_score.mean())
        self.sym_std_ = float(symbolic_score.std())

        symbolic_norm = self._normalize_symbolic(symbolic_score)

        best_lambda = 0.0
        best_value = -np.inf

        for lam in self.lambda_grid:
            candidate = np.clip(
                neural_score + float(lam) * symbolic_norm,
                0.0, 1.0
            )
            value = self._metric(y_val, candidate)
            if value > best_value:
                best_value = value
                best_lambda = float(lam)

        print(f"[Symbolic] Best lambda: {best_lambda:.4f} | Best {self.objective}: {best_value:.4f}")
        print(f"[Symbolic] Normalization params: mean={self.sym_mean_:.4f}, std={self.sym_std_:.4f}")

        self.lambda_ = best_lambda
        return self

    def score(self, neural_score: np.ndarray, symbolic_score: np.ndarray) -> tuple[np.ndarray, dict]:
        # CRITICAL: apply the SAME normalization used at fit time.
        symbolic_norm = self._normalize_symbolic(symbolic_score)
        combined = np.clip(
            neural_score + self.lambda_ * symbolic_norm,
            0.0, 1.0
        ).astype(np.float32)
        return combined, {"lambda": float(self.lambda_)}

    def state_dict(self) -> dict:
        return {
            "lambda": float(self.lambda_),
            "objective": self.objective,
            "sym_mean": float(self.sym_mean_),
            "sym_std": float(self.sym_std_),
        }
