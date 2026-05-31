from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


@dataclass
class ThresholdProfile:
    threshold: float
    scale: float
    direction: str


@dataclass
class RuleDefinition:
    name: str
    description: str
    feature_fn: Callable[[pd.DataFrame], np.ndarray]
    direction: str = 'high'
    support_floor: float = 0.05


@dataclass
class FittedRule:
    name: str
    description: str
    threshold: float
    scale: float
    direction: str
    weight: float
    feature_fn: Callable[[pd.DataFrame], np.ndarray]


def _safe_array(x: Iterable[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def syn_flood_signal(df: pd.DataFrame) -> np.ndarray:
    return _safe_array(
        0.40 * df['serror_rate']
        + 0.30 * df['srv_serror_rate']
        + 0.20 * df['dst_host_serror_rate']
        + 0.10 * np.clip(df['count'] / 300.0, 0.0, 1.0)
    )


def service_sweep_signal(df: pd.DataFrame) -> np.ndarray:
    return _safe_array(
        0.45 * df['diff_srv_rate']
        + 0.30 * (1.0 - df['same_srv_rate'])
        + 0.25 * df['srv_diff_host_rate']
    )


def low_entropy_repeat_signal(df: pd.DataFrame) -> np.ndarray:
    src = np.log1p(df['src_bytes'].to_numpy(dtype=np.float32))
    dst = np.log1p(df['dst_bytes'].to_numpy(dtype=np.float32))
    ratio = np.clip(src / np.maximum(dst, 1e-3), 0.0, 8.0)
    return _safe_array(
        0.40 * np.clip((ratio - 1.0) / 4.0, 0.0, 1.0)
        + 0.35 * np.clip(df['dst_host_srv_count'] / 255.0, 0.0, 1.0)
        + 0.25 * (1.0 - np.clip(df['dst_host_diff_srv_rate'], 0.0, 1.0))
    )


def auth_compromise_signal(df: pd.DataFrame) -> np.ndarray:
    return _safe_array(
        0.35 * np.clip(df['num_failed_logins'] / 5.0, 0.0, 1.0)
        + 0.30 * np.clip(df['num_compromised'] / 5.0, 0.0, 1.0)
        + 0.20 * np.clip(df['root_shell'], 0.0, 1.0)
        + 0.15 * np.clip(df['is_guest_login'], 0.0, 1.0)
    )


def default_rule_definitions() -> list[RuleDefinition]:
    return [
        RuleDefinition(
            name='syn_flood_consensus',
            description='Elevated SYN-style error rates and repetitive connection pressure.',
            feature_fn=syn_flood_signal,
            direction='high',
        ),
        RuleDefinition(
            name='service_sweep_dispersion',
            description='High service dispersion with low same-service continuity.',
            feature_fn=service_sweep_signal,
            direction='high',
        ),
        RuleDefinition(
            name='low_entropy_repetition',
            description='Repeated low-diversity traffic bursts suggestive of scripted probing.',
            feature_fn=low_entropy_repeat_signal,
            direction='high',
        ),
        RuleDefinition(
            name='auth_compromise_pattern',
            description='Authentication failures and compromise indicators above normal profile.',
            feature_fn=auth_compromise_signal,
            direction='high',
        ),
    ]


def fit_rule_profile(values_normal: np.ndarray, values_attack: np.ndarray, direction: str = 'high') -> ThresholdProfile:
    values_normal = _safe_array(values_normal)
    values_attack = _safe_array(values_attack)

    if direction == 'high':
        q_normal = float(np.quantile(values_normal, 0.95))
        q_attack = float(np.quantile(values_attack, 0.50)) if len(values_attack) else q_normal
        threshold = max(q_normal, 0.5 * (q_normal + q_attack))
        spread = float(max(np.quantile(values_normal, 0.75) - np.quantile(values_normal, 0.25), 1e-3))
    else:
        q_normal = float(np.quantile(values_normal, 0.05))
        q_attack = float(np.quantile(values_attack, 0.50)) if len(values_attack) else q_normal
        threshold = min(q_normal, 0.5 * (q_normal + q_attack))
        spread = float(max(np.quantile(values_normal, 0.75) - np.quantile(values_normal, 0.25), 1e-3))

    return ThresholdProfile(threshold=threshold, scale=spread, direction=direction)


def apply_rule_profile(values: np.ndarray, profile: ThresholdProfile) -> np.ndarray:
    values = _safe_array(values)
    if profile.direction == 'high':
        margin = (values - profile.threshold) / profile.scale
    else:
        margin = (profile.threshold - values) / profile.scale
    return _sigmoid(margin)
