from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable
import copy, random, os
import numpy as np, torch, yaml

NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]
CATEGORICAL_COLUMNS = ['protocol_type', 'service', 'flag']
META_COLUMNS = ['label', 'difficulty', 'raw_label', 'attack_family', 'target']
ATTACK_FAMILY_MAP = {
    'normal': 'Normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L',
    'snmpguess': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'httptunnel': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}
@dataclass
class ExperimentPaths:
    root: Path
    experiment_dir: Path
    plots_dir: Path
    reports_dir: Path
    models_dir: Path

def read_yaml(path: str|Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as h: return yaml.safe_load(h) or {}

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged=copy.deepcopy(base)
    for k,v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict): merged[k]=deep_merge(merged[k], v)
        else: merged[k]=copy.deepcopy(v)
    return merged

def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    if value is None: return
    keys=dotted_key.split('.')
    cur=cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict): cur[k]={}
        cur=cur[k]
    cur[keys[-1]]=value

def load_experiment_config(base_path: str|Path, override_paths: Iterable[str|Path]|None=None, cli_overrides: Dict[str,Any]|None=None) -> Dict[str,Any]:
    cfg=read_yaml(base_path)
    for p in override_paths or []: cfg=deep_merge(cfg, read_yaml(p))
    for k,v in (cli_overrides or {}).items(): set_nested(cfg,k,v)
    cfg.setdefault('runtime', {})
    cfg['runtime'].setdefault('seed', 42)
    cfg['runtime'].setdefault('smoke_test', False)
    cfg['runtime'].setdefault('raw_data_dir', 'data/raw')
    cfg['runtime'].setdefault('artifacts_dir', 'artifacts')
    cfg.setdefault('dataset', {}).setdefault('version','NSL-KDD')
    cfg.setdefault('methods', {}).setdefault('clafpp', {})
    return cfg

def make_experiment_paths(artifacts_root: str|Path, explicit_experiment_id: str|None=None) -> ExperimentPaths:
    root=Path(artifacts_root)/'experiments'; root.mkdir(parents=True, exist_ok=True)
    if explicit_experiment_id: exp_id=explicit_experiment_id
    else:
        ids=[]
        for p in root.glob('exp_*'):
            try: ids.append(int(p.name.split('_')[-1]))
            except: pass
        exp_id=f'exp_{max(ids, default=0)+1:03d}'
    exp=root/exp_id; plots=exp/'plots'; reports=exp/'reports'; models=exp/'models'
    for p in [exp,plots,reports,models]: p.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(root=root, experiment_dir=exp, plots_dir=plots, reports_dir=reports, models_dir=models)

def set_global_seed(seed:int=42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    try: torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    except: pass
