from __future__ import annotations
import argparse
from pathlib import Path
from src.config import load_experiment_config
from src.pipeline import run_pipeline

def parse_args():
    p=argparse.ArgumentParser(description="CLAF++: Conditional Latent Anomaly Fusion with Ensemble Intelligence")
    p.add_argument("--config",type=str,default="configs/base.yaml")
    p.add_argument("--override",action="append",default=[])
    p.add_argument("--raw-data-dir",type=str,default=None)
    p.add_argument("--artifacts-dir",type=str,default=None)
    p.add_argument("--seed",type=int,default=None)
    p.add_argument("--smoke-test",action="store_true")
    p.add_argument("--experiment-id",type=str,default=None)
    return p.parse_args()

def main():
    a=parse_args()
    cfg=load_experiment_config(a.config, a.override, {
        "runtime.seed": a.seed,
        "runtime.raw_data_dir": a.raw_data_dir,
        "runtime.artifacts_dir": a.artifacts_dir,
        "runtime.smoke_test": True if a.smoke_test else None,
        "runtime.experiment_id": a.experiment_id,
    })
    table, metadata = run_pipeline(cfg)
    print("\nCLAF++ Research Pipeline\n")
    print(table.to_string(index=False))
    print(f"\nExperiment directory: {Path(metadata['experiment_dir']).resolve()}")
    print(f"Predictions file: {Path(metadata['experiment_dir']) / 'predictions.csv'}")

if __name__ == "__main__":
    main()
