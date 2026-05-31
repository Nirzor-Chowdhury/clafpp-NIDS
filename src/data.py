from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from .config import ATTACK_FAMILY_MAP, CATEGORICAL_COLUMNS, META_COLUMNS, NSL_KDD_COLUMNS

def _normalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    df['raw_label']=df['label'].astype(str).str.strip().str.lower()
    df['attack_family']=df['raw_label'].map(ATTACK_FAMILY_MAP).fillna('OtherAttack')
    df['target']=(df['raw_label']!='normal').astype(int)
    df['su_attempted']=df['su_attempted'].replace(2,1)
    df['difficulty']=pd.to_numeric(df['difficulty'], errors='coerce').fillna(0).astype(int)
    for col in CATEGORICAL_COLUMNS: df[col]=df[col].astype(str).str.strip().str.lower()
    return df

def load_nsl_kdd(train_path: str|Path, test_path: str|Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df=pd.read_csv(train_path, names=NSL_KDD_COLUMNS)
    test_df=pd.read_csv(test_path, names=NSL_KDD_COLUMNS)
    return _normalize_common_columns(train_df), _normalize_common_columns(test_df)

def split_train_validation(train_df: pd.DataFrame, val_size: float, seed: int):
    train_idx, val_idx = train_test_split(train_df.index, test_size=val_size, random_state=seed, stratify=train_df['target'])
    return train_df.loc[train_idx].reset_index(drop=True), train_df.loc[val_idx].reset_index(drop=True)

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]

def make_smoke_test_frames(n_train: int=300, n_test: int=120, seed: int=42):
    rng=np.random.default_rng(seed)
    services=['http','smtp','ftp','private','domain_u','eco_i']
    flags=['sf','rej','s0','rstr']
    protocols=['tcp','udp','icmp']
    def build(n:int):
        y=rng.binomial(1,0.40,size=n); rows=[]
        for yi in y:
            is_attack=yi==1; family='Normal' if yi==0 else rng.choice(['DoS','Probe','R2L','U2R'],p=[0.55,0.23,0.16,0.06]); raw='normal' if yi==0 else family.lower()
            rows.append({
                'duration': rng.gamma(1.8+1.4*is_attack,4.1), 'protocol_type': rng.choice(protocols,p=[0.66,0.20,0.14] if not is_attack else [0.51,0.17,0.32]), 'service': rng.choice(services), 'flag': rng.choice(flags,p=[0.72,0.10,0.12,0.06] if not is_attack else [0.45,0.18,0.28,0.09]),
                'src_bytes': rng.lognormal(6.5 if not is_attack else 7.4,1.0), 'dst_bytes': rng.lognormal(6.2 if not is_attack else 5.6,1.1), 'land': rng.binomial(1,0.002 if not is_attack else 0.03), 'wrong_fragment': rng.poisson(0.02 if not is_attack else 0.18), 'urgent': rng.poisson(0.002 if not is_attack else 0.03), 'hot': rng.poisson(0.15 if not is_attack else 1.4),
                'num_failed_logins': rng.poisson(0.03 if not is_attack else 0.45), 'logged_in': rng.binomial(1,0.78 if not is_attack else 0.38), 'num_compromised': rng.poisson(0.04 if not is_attack else 1.30), 'root_shell': rng.binomial(1,0.001 if not is_attack else 0.04), 'su_attempted': rng.binomial(1,0.001 if not is_attack else 0.03), 'num_root': rng.poisson(0.02 if not is_attack else 0.75),
                'num_file_creations': rng.poisson(0.04 if not is_attack else 0.38), 'num_shells': rng.poisson(0.001 if not is_attack else 0.05), 'num_access_files': rng.poisson(0.03 if not is_attack else 0.33), 'num_outbound_cmds':0, 'is_host_login':0, 'is_guest_login': rng.binomial(1,0.02 if not is_attack else 0.10),
                'count': rng.integers(0,160 if not is_attack else 320), 'srv_count': rng.integers(0,120 if not is_attack else 280), 'serror_rate': np.clip(rng.normal(0.10 if not is_attack else 0.58,0.15),0,1), 'srv_serror_rate': np.clip(rng.normal(0.09 if not is_attack else 0.61,0.15),0,1), 'rerror_rate': np.clip(rng.normal(0.04 if not is_attack else 0.31,0.12),0,1), 'srv_rerror_rate': np.clip(rng.normal(0.04 if not is_attack else 0.29,0.12),0,1),
                'same_srv_rate': np.clip(rng.normal(0.72 if not is_attack else 0.41,0.16),0,1), 'diff_srv_rate': np.clip(rng.normal(0.10 if not is_attack else 0.34,0.12),0,1), 'srv_diff_host_rate': np.clip(rng.normal(0.07 if not is_attack else 0.23,0.10),0,1), 'dst_host_count': rng.integers(0,255), 'dst_host_srv_count': rng.integers(0,255), 'dst_host_same_srv_rate': np.clip(rng.normal(0.69 if not is_attack else 0.44,0.15),0,1), 'dst_host_diff_srv_rate': np.clip(rng.normal(0.11 if not is_attack else 0.29,0.12),0,1),
                'dst_host_same_src_port_rate': np.clip(rng.normal(0.60 if not is_attack else 0.33,0.18),0,1), 'dst_host_srv_diff_host_rate': np.clip(rng.normal(0.06 if not is_attack else 0.16,0.10),0,1), 'dst_host_serror_rate': np.clip(rng.normal(0.11 if not is_attack else 0.63,0.14),0,1), 'dst_host_srv_serror_rate': np.clip(rng.normal(0.10 if not is_attack else 0.65,0.14),0,1), 'dst_host_rerror_rate': np.clip(rng.normal(0.05 if not is_attack else 0.27,0.12),0,1), 'dst_host_srv_rerror_rate': np.clip(rng.normal(0.05 if not is_attack else 0.25,0.11),0,1),
                'label': raw, 'difficulty': int(rng.integers(1,22)),
            })
        return _normalize_common_columns(pd.DataFrame(rows))
    return build(n_train), build(n_test)
