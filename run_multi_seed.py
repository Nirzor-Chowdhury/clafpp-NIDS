"""
Multi-seed experiment harness for CLAFPP++.

Runs the pipeline across N seeds on a given config, collects per-seed
leaderboards, computes mean ± std per (model, metric), and runs Wilcoxon
paired tests between every model and a chosen baseline.

Usage:
    python run_multi_seed.py --config configs/base.yaml --seeds 42 7 123 --tag nslkdd
    python run_multi_seed.py --config configs/base_edge.yaml --seeds 42 7 123 --tag edge
"""

from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon, ttest_rel

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import run_pipeline


def cohens_d(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size < 2 or diffs.std(ddof=1) == 0:
        return 0.0
    return float(diffs.mean() / diffs.std(ddof=1))


def effect_label(d: float) -> str:
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123])
    ap.add_argument('--tag', required=True, help='Output folder tag, e.g. nslkdd or edge')
    ap.add_argument('--baseline', default='Ensemble (no meta)',
                    help='Reference model for paired statistical tests')
    args = ap.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    out_root = Path('artifacts') / 'multiseed' / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    per_seed_tables = []

    for seed in args.seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        cfg = copy.deepcopy(base_cfg)
        cfg['runtime']['seed'] = int(seed)
        cfg['runtime']['experiment_id'] = f'{args.tag}_seed{seed}'

        table, metadata = run_pipeline(cfg)
        df = pd.DataFrame(table)
        df['seed'] = int(seed)
        per_seed_tables.append(df)

        df.to_csv(out_root / f'leaderboard_seed{seed}.csv', index=False)

    all_df = pd.concat(per_seed_tables, ignore_index=True)
    all_df.to_csv(out_root / 'all_seeds_raw.csv', index=False)

    # Aggregate: mean and std per (Model, metric)
    metric_cols = ['AUROC', 'AUPRC', 'F1', 'Precision', 'Recall', 'FalseAlarmRate']
    metric_cols = [c for c in metric_cols if c in all_df.columns]

    agg = all_df.groupby('Model')[metric_cols].agg(['mean', 'std']).round(4)
    agg.to_csv(out_root / 'aggregated_mean_std.csv')

    print(f"\n\n{'='*60}\nAGGREGATED (mean ± std) — {args.tag}\n{'='*60}")
    print(agg.to_string())

    # Paired statistical tests vs. baseline
    baseline = args.baseline
    if baseline not in all_df['Model'].unique():
        print(f"\n⚠ baseline '{baseline}' not found; available: {sorted(all_df['Model'].unique())}")
        return

    print(f"\n\n{'='*60}\nPAIRED STATISTICAL TESTS vs. {baseline}\n{'='*60}")

    base_rows = all_df[all_df['Model'] == baseline].sort_values('seed')
    rows = []
    for model in sorted(all_df['Model'].unique()):
        if model == baseline:
            continue
        model_rows = all_df[all_df['Model'] == model].sort_values('seed')
        # F1 paired test
        a = model_rows['F1'].values
        b = base_rows['F1'].values
        diffs = a - b
        # Wilcoxon needs >= 1 non-zero diff; protect against tiny samples
        try:
            if np.all(diffs == 0) or len(diffs) < 2:
                p_w = float('nan')
            else:
                _, p_w = wilcoxon(a, b, zero_method='zsplit')
        except Exception:
            p_w = float('nan')
        try:
            _, p_t = ttest_rel(a, b)
        except Exception:
            p_t = float('nan')
        d = cohens_d(diffs)
        rows.append({
            'Model': model,
            'F1_mean_diff': round(float(diffs.mean()), 4),
            'F1_std_diff': round(float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0, 4),
            'Wilcoxon_p': round(p_w, 4) if not np.isnan(p_w) else 'n/a',
            'Paired_t_p': round(p_t, 4) if not np.isnan(p_t) else 'n/a',
            'Cohens_d': round(d, 3),
            'Effect': effect_label(d),
            'Sig_at_0.05': 'YES' if (not np.isnan(p_w) and p_w < 0.05) else 'NO',
        })
    stat_df = pd.DataFrame(rows)
    print(stat_df.to_string(index=False))
    stat_df.to_csv(out_root / 'paired_tests_vs_baseline.csv', index=False)

    summary = {
        'config': args.config,
        'seeds': args.seeds,
        'tag': args.tag,
        'baseline': baseline,
        'n_models': int(all_df['Model'].nunique()),
        'metrics_reported': metric_cols,
    }
    with open(out_root / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ All artifacts written to {out_root}/")


if __name__ == '__main__':
    main()