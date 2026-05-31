"""
Tuned XGBoost-alone baseline for CLAFPP++ comparison.

Runs a small stratified-CV hyperparameter search per seed, then evaluates on
the same test set as the main pipeline, using the same evaluator and threshold
selection. Aggregates mean+/-std across seeds for direct comparison with the
CLAF++ leaderboard.

Usage:
    python run_xgb_baseline.py --config configs/base.yaml --seeds 42 7 123 1 99 --tag nslkdd
    python run_xgb_baseline.py --config configs/base_edge.yaml --seeds 42 7 123 1 99 --tag edge
"""

from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon, ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from src.training.trainer import ResearchTrainer
from src.evaluation.evaluator import ResearchEvaluator


# Small, defensible grid. 3 values per axis -> 81 combinations,
# evaluated with 3-fold CV = 243 fits per seed. Manageable.
PARAM_GRID = {
    'n_estimators':       [200, 400, 800],
    'max_depth':          [4, 6, 8],
    'learning_rate':      [0.03, 0.07, 0.15],
    'min_child_weight':   [1, 3, 7],
}


def cohens_d(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size < 2 or diffs.std(ddof=1) == 0:
        return 0.0
    return float(diffs.mean() / diffs.std(ddof=1))


def grid_iter(grid: dict):
    keys = list(grid.keys())
    sizes = [len(grid[k]) for k in keys]
    n = int(np.prod(sizes))
    for i in range(n):
        cfg = {}
        idx = i
        for k, s in zip(keys, sizes):
            cfg[k] = grid[k][idx % s]
            idx //= s
        yield cfg


def cv_search(X, y, seed: int, cv_splits: int = 3) -> dict:
    """Stratified K-fold grid search. Returns best params + best CV AUROC."""
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    best_params = None
    best_score = -np.inf

    combos = list(grid_iter(PARAM_GRID))
    total = len(combos)
    for i, params in enumerate(combos, 1):
        fold_aucs = []
        for tr_idx, va_idx in skf.split(X, y):
            clf = XGBClassifier(
                **params,
                objective='binary:logistic',
                eval_metric='logloss',
                tree_method='hist',
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
            )
            clf.fit(X[tr_idx], y[tr_idx])
            proba = clf.predict_proba(X[va_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y[va_idx], proba))
        score = float(np.mean(fold_aucs))
        if score > best_score:
            best_score = score
            best_params = dict(params)
        # progress beat every 20 combos
        if i % 20 == 0:
            print(f"    [grid] {i}/{total} combos | running best AUROC={best_score:.4f}")

    return {'params': best_params, 'cv_auroc': best_score}


def evaluate_on_test(X_tr, y_tr, X_te, y_te, params: dict, seed: int) -> dict:
    """Fit best params on full train, evaluate on test with project's evaluator."""
    clf = XGBClassifier(
        **params,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]

    ev = ResearchEvaluator(grid_size=101, recall_floor=0.85, far_ceiling=0.20)
    result, _ = ev.evaluate('Tuned XGBoost', y_te, proba)
    m = result.metrics
    return {
        'threshold': float(result.threshold),
        'AUROC': float(m['auroc']),
        'AUPRC': float(m['auprc']),
        'F1': float(m['f1']),
        'Precision': float(m['precision']),
        'Recall': float(m['recall']),
        'FalseAlarmRate': float(m['false_alarm_rate']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123, 1, 99])
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    print(f"\nXGBoost baseline | config={args.config} | seeds={args.seeds} | tag={args.tag}")
    sys.stdout.flush()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    out_root = Path('artifacts') / 'baselines' / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    per_seed = []

    for seed in args.seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        sys.stdout.flush()

        cfg = copy.deepcopy(base_cfg)
        cfg['runtime']['seed'] = int(seed)
        cfg['runtime']['experiment_id'] = f'xgb_baseline_{args.tag}_seed{seed}'

        # Use the project's trainer to get IDENTICAL splits/preprocessing as CLAF++.
        trainer = ResearchTrainer(cfg)
        p = trainer.prepare_data()

        # Train pool = train + val (single classifier gets all available
        # training data, fair comparison vs. the stacked CLAF++ pipeline).
        X_tr_full = np.vstack([p.X_train_std, p.X_val_std]).astype(np.float32)
        y_tr_full = np.concatenate([p.y_train, p.y_val]).astype(np.int64)
        X_te = p.X_test_std.astype(np.float32)
        y_te = p.y_test.astype(np.int64)

        print(f"  Training pool: {X_tr_full.shape} | test: {X_te.shape}")
        sys.stdout.flush()

        t = time.perf_counter()
        cv = cv_search(X_tr_full, y_tr_full, seed=seed, cv_splits=3)
        cv_time = time.perf_counter() - t
        print(f"  [done] CV search: {cv_time:.1f}s | best AUROC={cv['cv_auroc']:.4f}")
        print(f"         best params: {cv['params']}")

        t = time.perf_counter()
        result = evaluate_on_test(X_tr_full, y_tr_full, X_te, y_te, cv['params'], seed=seed)
        fit_time = time.perf_counter() - t
        result['seed'] = int(seed)
        result['cv_search_seconds'] = round(cv_time, 1)
        result['final_fit_seconds'] = round(fit_time, 1)
        result['best_params'] = cv['params']
        result['Model'] = 'Tuned XGBoost'

        print(f"  TEST: F1={result['F1']:.4f}  AUROC={result['AUROC']:.4f}  "
              f"FAR={result['FalseAlarmRate']:.4f}  threshold={result['threshold']:.3f}")
        sys.stdout.flush()

        per_seed.append(result)

        with open(out_root / f'seed{seed}.json', 'w') as f:
            json.dump(result, f, indent=2)

    df = pd.DataFrame(per_seed)
    df.to_csv(out_root / 'all_seeds.csv', index=False)

    metric_cols = ['AUROC', 'AUPRC', 'F1', 'Precision', 'Recall', 'FalseAlarmRate']
    agg = df[metric_cols].agg(['mean', 'std']).round(4)
    agg.to_csv(out_root / 'aggregated_mean_std.csv')

    print(f"\n\n{'='*60}\nTUNED XGBOOST - {args.tag} - aggregated\n{'='*60}")
    print(agg.to_string())

    # Compare to CLAF++ multi-seed if available
    claff_path = Path('artifacts') / 'multiseed' / args.tag / 'all_seeds_raw.csv'
    if claff_path.exists():
        print(f"\n{'='*60}\nPAIRED COMPARISONS vs CLAF++ multi-seed runs\n{'='*60}")
        claff_all = pd.read_csv(claff_path)
        for ref_model in ['Ensemble (no meta)', 'CLAF++ + Symbolic', 'RF/XGB']:
            ref = claff_all[claff_all['Model'] == ref_model].sort_values('seed')
            xgb = df.sort_values('seed')
            if len(ref) != len(xgb):
                continue
            a = xgb['F1'].values
            b = ref['F1'].values
            diffs = a - b
            try:
                _, p_w = wilcoxon(a, b, zero_method='zsplit')
            except Exception:
                p_w = float('nan')
            try:
                _, p_t = ttest_rel(a, b)
            except Exception:
                p_t = float('nan')
            d = cohens_d(diffs)
            print(f"\n  Tuned XGBoost vs {ref_model}:")
            print(f"    F1 mean diff: {diffs.mean():+.4f} (std {diffs.std(ddof=1):.4f})")
            print(f"    Wilcoxon p={p_w:.4f}  |  Paired t p={p_t:.4f}  |  Cohen's d={d:.3f}")

    print(f"\n[done] Artifacts written to {out_root}/")


if __name__ == '__main__':
    main()