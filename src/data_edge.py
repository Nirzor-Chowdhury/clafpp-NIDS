from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LEAKY_COLS = [
    'dns.qry.name.len',
    'mqtt.topic',
    'mqtt.conack.flags',
    'mqtt.msg',
    'mqtt.protoname',
    'tcp.dstport',
]

REDUNDANT_COLS = ['mqtt.conflag.cleansess', 'mqtt.hdrflags', 'mqtt.ver']

META_COLS_EDGE = ['Attack_label', 'Attack_type', 'is_vulnerable_trigger']


def _clean_edge(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in REDUNDANT_COLS + LEAKY_COLS if c in df.columns])
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    # Standardize target/label naming to match NSL-KDD pipeline expectations
    df['target'] = df['Attack_label'].astype(int)
    df['raw_label'] = df['Attack_type'].astype(str).str.lower()
    df['attack_family'] = df['Attack_type'].astype(str)
    df['difficulty'] = 0  # not present in Edge-IIoTset; harmless placeholder
    return df


def load_edge_iiotset(csv_path: str | Path, test_size: float = 0.25, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Edge-IIoTset and produce train/test frames matching the NSL-KDD interface."""
    df = pd.read_csv(csv_path)
    df = _clean_edge(df)
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['target'], random_state=seed
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_edge_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(META_COLS_EDGE + LEAKY_COLS + ['target', 'raw_label', 'attack_family', 'difficulty'])
    return [c for c in df.columns if c not in excluded]