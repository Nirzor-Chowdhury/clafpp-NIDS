from __future__ import annotations
import json
import pandas as pd, torch, yaml
from .config import ExperimentPaths
class ExperimentTracker:
    def __init__(self, paths: ExperimentPaths): self.paths=paths
    def save_config(self, config: dict):
        with open(self.paths.experiment_dir/'config.yaml','w',encoding='utf-8') as h: yaml.safe_dump(config,h,sort_keys=False)
    def save_json(self,payload: dict, relative_path: str):
        p=self.paths.experiment_dir/relative_path; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    def save_csv(self, frame: pd.DataFrame, relative_path: str):
        p=self.paths.experiment_dir/relative_path; p.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(p,index=False)
    def save_model_bundle(self, bundle: dict): torch.save(bundle, self.paths.experiment_dir/'model.pt')
