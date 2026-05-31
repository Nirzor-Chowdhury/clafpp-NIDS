from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .rules import FittedRule, ThresholdProfile, apply_rule_profile, default_rule_definitions, fit_rule_profile


@dataclass
class RuleEngineOutput:
    score: np.ndarray
    rule_frame: pd.DataFrame
    explanations: list[str]
    summary: list[dict]


class NeuroSymbolicRuleEngine:
    """Data-driven symbolic layer.

    Each rule learns a threshold from train traffic and a usefulness-based
    validation weight. That makes the symbolic layer defensible and reproducible.
    """

    def __init__(self, rules=None, weight_floor: float = 0.05):
        self.rules = rules or default_rule_definitions()
        self.weight_floor = float(weight_floor)
        self.fitted_rules_: list[FittedRule] = []

    def fit(self, train_df: pd.DataFrame, y_train: np.ndarray, val_df: pd.DataFrame | None = None, y_val: np.ndarray | None = None):
        fitted = []
        normal_mask = np.asarray(y_train) == 0
        attack_mask = np.asarray(y_train) == 1

        for rule in self.rules:
            train_signal = rule.feature_fn(train_df)
            profile = fit_rule_profile(train_signal[normal_mask], train_signal[attack_mask], direction=rule.direction)

            weight = rule.support_floor
            if val_df is not None and y_val is not None and len(np.unique(y_val)) > 1:
                val_signal = rule.feature_fn(val_df)
                val_confidence = apply_rule_profile(val_signal, profile)
                weight = max(float(average_precision_score(y_val, val_confidence)), rule.support_floor)

            fitted.append(
                FittedRule(
                    name=rule.name,
                    description=rule.description,
                    threshold=float(profile.threshold),
                    scale=float(profile.scale),
                    direction=rule.direction,
                    weight=float(weight),
                    feature_fn=rule.feature_fn,
                )
            )

        weight_sum = sum(r.weight for r in fitted)
        self.fitted_rules_ = [
            FittedRule(
                name=r.name,
                description=r.description,
                threshold=r.threshold,
                scale=r.scale,
                direction=r.direction,
                weight=float(r.weight / weight_sum),
                feature_fn=r.feature_fn,
            )
            for r in fitted
        ]
        return self

    def evaluate(self, df: pd.DataFrame, neural_signals: dict | None = None) -> RuleEngineOutput:
        if not self.fitted_rules_:
            raise RuntimeError('Rule engine must be fit before evaluate().')

        columns = {}
        summary = []
        for rule in self.fitted_rules_:
            raw_values = rule.feature_fn(df)
            confidence = apply_rule_profile(
                raw_values,
                ThresholdProfile(threshold=rule.threshold, scale=rule.scale, direction=rule.direction),
            )
            columns[rule.name] = confidence
            summary.append(
                {
                    'rule': rule.name,
                    'description': rule.description,
                    'threshold': float(rule.threshold),
                    'scale': float(rule.scale),
                    'weight': float(rule.weight),
                    'mean_confidence': float(confidence.mean()),
                    'trigger_rate_0_7': float((confidence >= 0.7).mean()),
                }
            )

        frame = pd.DataFrame(columns)
        # 🔥 USE LEARNED RULES (CORRECT WAY)
        weight_vec = np.array([r.weight for r in self.fitted_rules_], dtype=np.float32)

        rule_scores = frame.values.astype(np.float32)

        score = (rule_scores * weight_vec).sum(axis=1)
        score = score / (score.max() + 1e-6)

        explanations = []
        for _, row in frame.iterrows():
            active = []
            for rule in self.fitted_rules_:
                value = float(row[rule.name])
                if value >= 0.55:
                    active.append((rule.name, value))
            active.sort(key=lambda x: x[1], reverse=True)
            explanations.append('; '.join([f'{name}:{value:.2f}' for name, value in active[:3]]) if active else 'no-symbolic-trigger')

        return RuleEngineOutput(score=score, rule_frame=frame, explanations=explanations, summary=summary)
