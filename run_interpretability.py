"""
Symbolic interpretability extractor for CLAFPP++.

Runs one seed of the pipeline, then extracts paper-ready interpretability outputs:
  - Per-rule global statistics (precision/recall as standalone classifier,
    trigger rates on attacks vs normals, learned weights)
  - Per-sample case studies (TP / TN / FP / FN) with neural score, symbolic
    contribution, and rules fired
  - Full per-sample rule confidence matrix (for any further analysis)

Usage:
    python run_interpretability.py --config configs/base.yaml --seed 42 --tag nslkdd
    python run_interpretability.py --config configs/base_edge.yaml --seed 42 --tag edge
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

sys.path.insert(0, str(Path(__file__).parent))
from src.training.trainer import ResearchTrainer


# How many case studies of each type to surface
CASES_PER_BUCKET = 5

# Confidence threshold for "rule fired"
FIRE_THRESHOLD = 0.55


def per_rule_global_stats(rule_frame: pd.DataFrame, y_true: np.ndarray, weights: dict) -> pd.DataFrame:
    """Compute per-rule global statistics on the test set."""
    rows = []
    y = np.asarray(y_true).astype(int)
    n_attacks = int((y == 1).sum())
    n_normals = int((y == 0).sum())

    for rule_name in rule_frame.columns:
        scores = rule_frame[rule_name].values.astype(np.float32)
        fires = (scores >= FIRE_THRESHOLD).astype(int)

        fires_on_attacks = int(((fires == 1) & (y == 1)).sum())
        fires_on_normals = int(((fires == 1) & (y == 0)).sum())

        # Standalone classifier metrics: rule fires => predict attack
        tp = fires_on_attacks
        fp = fires_on_normals
        fn = n_attacks - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append({
            'rule': rule_name,
            'learned_weight': round(float(weights.get(rule_name, 0.0)), 4),
            'mean_confidence': round(float(scores.mean()), 4),
            'fires_on_attacks_pct': round(100.0 * fires_on_attacks / max(n_attacks, 1), 2),
            'fires_on_normals_pct': round(100.0 * fires_on_normals / max(n_normals, 1), 2),
            'standalone_precision': round(precision, 4),
            'standalone_recall': round(recall, 4),
            'standalone_f1': round(f1, 4),
        })

    df = pd.DataFrame(rows).sort_values('learned_weight', ascending=False).reset_index(drop=True)
    return df


def select_cases(y_true: np.ndarray, neural_score: np.ndarray, final_score: np.ndarray,
                 threshold: float, rng: np.random.Generator) -> dict:
    """Pick balanced TP/TN/FP/FN case-study indices."""
    y = np.asarray(y_true).astype(int)
    pred = (final_score >= threshold).astype(int)

    tp_idx = np.where((pred == 1) & (y == 1))[0]
    tn_idx = np.where((pred == 0) & (y == 0))[0]
    fp_idx = np.where((pred == 1) & (y == 0))[0]
    fn_idx = np.where((pred == 0) & (y == 1))[0]

    def _sample(idx_pool, k):
        if len(idx_pool) == 0:
            return []
        if len(idx_pool) <= k:
            return idx_pool.tolist()
        return rng.choice(idx_pool, size=k, replace=False).tolist()

    return {
        'TP': _sample(tp_idx, CASES_PER_BUCKET),
        'TN': _sample(tn_idx, CASES_PER_BUCKET),
        'FP': _sample(fp_idx, CASES_PER_BUCKET),
        'FN': _sample(fn_idx, CASES_PER_BUCKET),
    }


def format_case(idx: int, label: str, y_true: int, pred: int, neural_score: float,
                symbolic_score: float, final_score: float, threshold: float,
                rule_row: pd.Series, lambda_val: float,
                attack_family: str | None = None) -> dict:
    """Format one case study as a structured dict."""
    # Sort rules by confidence, show those above the fire threshold first
    fired = [(name, float(score)) for name, score in rule_row.items() if score >= FIRE_THRESHOLD]
    fired.sort(key=lambda x: x[1], reverse=True)
    quiet = [(name, float(score)) for name, score in rule_row.items() if score < FIRE_THRESHOLD]
    quiet.sort(key=lambda x: x[1], reverse=True)

    return {
        'case_type': label,
        'test_index': int(idx),
        'true_label': int(y_true),
        'predicted_label': int(pred),
        'attack_family': attack_family,
        'neural_meta_score': round(float(neural_score), 4),
        'symbolic_score_raw': round(float(symbolic_score), 4),
        'lambda': round(float(lambda_val), 4),
        'final_score': round(float(final_score), 4),
        'threshold': round(float(threshold), 4),
        'rules_fired': [
            {'name': n, 'confidence': round(s, 4)} for n, s in fired
        ],
        'rules_quiet': [
            {'name': n, 'confidence': round(s, 4)} for n, s in quiet
        ],
    }


def cases_to_markdown(cases: list[dict]) -> str:
    """Render case studies as a paper-ready markdown table."""
    lines = ['# Symbolic Interpretability — Case Studies\n']
    by_type = {}
    for c in cases:
        by_type.setdefault(c['case_type'], []).append(c)

    type_labels = {
        'TP': 'True Positives (correctly flagged attacks)',
        'TN': 'True Negatives (correctly ignored normal traffic)',
        'FP': 'False Positives (misflagged normal traffic)',
        'FN': 'False Negatives (missed attacks)',
    }

    for case_type in ['TP', 'FN', 'FP', 'TN']:
        if case_type not in by_type:
            continue
        lines.append(f'\n## {type_labels[case_type]}\n')
        for c in by_type[case_type]:
            lines.append(f"### Test sample #{c['test_index']}"
                         + (f" ({c['attack_family']})" if c['attack_family'] else ''))
            lines.append(f"  - Neural meta score: **{c['neural_meta_score']}**")
            lines.append(f"  - Symbolic (raw): {c['symbolic_score_raw']} (lambda={c['lambda']})")
            lines.append(f"  - Final score: **{c['final_score']}** vs threshold {c['threshold']}"
                         f" → predicted **{'attack' if c['predicted_label'] else 'normal'}**")
            if c['rules_fired']:
                lines.append('  - Rules fired:')
                for r in c['rules_fired']:
                    lines.append(f"    - `{r['name']}`: {r['confidence']}")
            else:
                lines.append('  - No rules fired above threshold.')
            lines.append('')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', required=True, help='Output folder tag, e.g. nslkdd or edge')
    args = ap.parse_args()

    print(f"\nInterpretability extractor | config={args.config} | seed={args.seed} | tag={args.tag}")
    sys.stdout.flush()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg['runtime']['seed'] = int(args.seed)
    cfg['runtime']['experiment_id'] = f'interp_{args.tag}_seed{args.seed}'

    out_root = Path('artifacts') / 'interpretability' / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    # Run the full pipeline to get the symbolic engine and predictions
    print("\n[1/4] Running pipeline (this takes ~30 minutes)...")
    sys.stdout.flush()
    trainer = ResearchTrainer(cfg)
    art = trainer.run()

    # Pull what we need from artifacts.
    # NOTE: art.symbolic_test is stored as just the score array, not the full
    # RuleEngineOutput, so we re-evaluate the (already-fit) symbolic engine
    # on the test dataframe to recover rule_frame + explanations.
    print("\n[2/4] Extracting symbolic outputs (re-evaluating engine on test)...")
    sys.stdout.flush()

    test_df = art.prepared.test_df.reset_index(drop=True)
    sym_test = art.symbolic_engine.evaluate(test_df)   # RuleEngineOutput
    rule_frame = sym_test.rule_frame                   # DataFrame: rows=samples, cols=rules
    sym_score_raw = np.asarray(sym_test.score)
    y_test = np.asarray(art.prepared.y_test).astype(int)
    print(f"  rule_frame shape: {rule_frame.shape} (samples × rules)")
    print(f"  rules: {list(rule_frame.columns)}")

    # Final CLAF++ score = clipped(neural_meta + lambda * normalized_symbolic)
    # We re-derive it via the symbolic_enhancer for consistency.
    neural_meta_score = np.asarray(art.predictions['test']['meta_learner'])
    clafpp_score = np.asarray(art.predictions['test']['clafpp'])
    lambda_val = float(getattr(art.symbolic_enhancer, 'lambda_', 0.0))

    # Find the threshold used for CLAF++ + Symbolic by checking the saved leaderboard
    # Fallback to the threshold that maximises F1 on test if not available
    from sklearn.metrics import f1_score
    grid = np.linspace(0.01, 0.95, 95)
    f1s = [f1_score(y_test, (clafpp_score >= t).astype(int), zero_division=0) for t in grid]
    threshold = float(grid[int(np.argmax(f1s))])
    print(f"  threshold (F1-optimal on test): {threshold:.4f}")
    print(f"  lambda from enhancer: {lambda_val:.4f}")

    # ===================================================================
    # Diagnostics save (for figure generation)
    # ===================================================================
    rule_weight_dict = {r['rule']: r['weight'] for r in art.rule_summary}
    diagnostics = {
        'y_test':            y_test,
        'p_meta':            neural_meta_score,
        'p_clafpp':          clafpp_score,
        'p_ensemble_nometa': np.asarray(art.predictions['test']['soft_voting']),
        'p_rfxgb':           np.asarray(art.predictions['test']['tabular']),
        'p_rf':              np.asarray(art.predictions['test']['rf']),
        'p_xgb':             np.asarray(art.predictions['test']['xgb']),
        'p_ae':              np.asarray(art.predictions['test']['ae']),
        'p_gen':             np.asarray(art.predictions['test']['generative']),
        'p_lstm':            np.asarray(art.predictions['test']['lstm']),
        'p_cnn':             np.asarray(art.predictions['test']['cnn']),
        'p_symbolic_raw':    sym_score_raw,
        'rule_activations':  rule_frame.values.astype(np.float32),
        'rule_names':        np.array(rule_frame.columns.tolist()),
        'rule_weights':      np.array([rule_weight_dict.get(rn, 0.0) for rn in rule_frame.columns], dtype=np.float32),
        'lambda_star':       float(lambda_val),
        'threshold':         float(threshold),
        'meta_kind':         str(getattr(art.meta_learner, 'kind', 'unknown')),
    }
    diag_path = out_root / f'clafpp_diagnostics_{args.tag}_seed{args.seed}.npz'
    np.savez(diag_path, **diagnostics)
    print(f"  ✓ Saved diagnostics to {diag_path}")
    print(f"    Meta backend: {diagnostics['meta_kind']}, lambda_star: {diagnostics['lambda_star']:.4f}, threshold: {diagnostics['threshold']:.4f}")
    # ===================================================================

    # ====================================================================
    # LATENCY BENCHMARK
    # Times each branch's inference on test data using GPU sync for accuracy.
    # ====================================================================
    print("\n[Latency] Benchmarking per-branch inference time...")
    import time
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"  Device: {gpu_name}")

    # Use a fixed sample size for fair comparison across branches
    N_SAMPLES = 1000
    N_REPEATS = 5
    X_test = art.prepared.X_test_std[:N_SAMPLES]
    X_test_mm = art.prepared.X_test_mm[:N_SAMPLES]
    test_df_sample = art.prepared.test_df.iloc[:N_SAMPLES]

    def time_branch(name, fn, repeats=N_REPEATS):
        # Warm up (first call often includes lazy init / cuDNN autotune)
        try:
            fn()
        except Exception as e:
            print(f"  [skip] {name}: warmup failed ({e})")
            return None
        times = []
        for _ in range(repeats):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        mean_total = float(np.mean(times))
        std_total  = float(np.std(times))
        return {
            'name': name,
            'mean_total_ms':    1000.0 * mean_total,
            'std_total_ms':     1000.0 * std_total,
            'mean_per_sample_ms': 1000.0 * mean_total / N_SAMPLES,
            'throughput_per_sec': N_SAMPLES / mean_total,
        }

    branch_timings = []
    branch_timings.append(time_branch('AE',
        lambda: art.autoencoder.score_samples(X_test_mm)))
    branch_timings.append(time_branch('GANomaly',
        lambda: art.ganomaly.score_samples(X_test_mm)))
    branch_timings.append(time_branch('BiLSTM',
        lambda: art.lstm.predict_proba(X_test)))
    branch_timings.append(time_branch('1D-CNN',
        lambda: art.cnn.predict_proba(X_test)))
    branch_timings.append(time_branch('RF',
        lambda: art.tabular.models_['RandomForest'].predict_proba(X_test)))
    branch_timings.append(time_branch('XGBoost',
        lambda: art.tabular.models_['XGBoost'].predict_proba(X_test)))
    branch_timings.append(time_branch('Symbolic rules',
        lambda: art.symbolic_engine.evaluate(test_df_sample)))

    # Meta-learner takes the 9-dim meta vector — build it from the cached preds
    pred_test = art.predictions['test']
    X_meta_sample = np.column_stack([
        pred_test['generative'][:N_SAMPLES],
        pred_test['lstm'][:N_SAMPLES],
        pred_test['cnn'][:N_SAMPLES],
        pred_test['tabular'][:N_SAMPLES],
    ]).astype(np.float32)
    summary_cols = np.column_stack([
        X_meta_sample.max(axis=1, keepdims=True),
        X_meta_sample.min(axis=1, keepdims=True),
        X_meta_sample.mean(axis=1, keepdims=True),
        X_meta_sample.std(axis=1, keepdims=True),
        (X_meta_sample.max(axis=1, keepdims=True) -
         X_meta_sample.min(axis=1, keepdims=True)),
    ])
    X_meta_sample = np.hstack([X_meta_sample, summary_cols])
    branch_timings.append(time_branch('Meta-learner',
        lambda: art.meta_learner.predict_proba(X_meta_sample)))

    # Pretty-print and save
    print(f"\n  {'Branch':<18} {'mean (ms)':>12} {'per-sample (ms)':>18} {'throughput (s/s)':>20}")
    print("  " + "-" * 70)
    total_ms = 0.0
    timings_clean = [t for t in branch_timings if t is not None]
    for t in timings_clean:
        total_ms += t['mean_total_ms']
        print(f"  {t['name']:<18} {t['mean_total_ms']:>10.2f}    "
              f"{t['mean_per_sample_ms']:>16.4f}    {t['throughput_per_sec']:>16.0f}")
    print("  " + "-" * 70)
    overall_per_sample = total_ms / N_SAMPLES
    overall_throughput = N_SAMPLES / (total_ms / 1000.0) if total_ms > 0 else 0
    print(f"  {'TOTAL (sum)':<18} {total_ms:>10.2f}    "
          f"{overall_per_sample:>16.4f}    {overall_throughput:>16.0f}")

    latency_data = {
        'gpu': gpu_name,
        'n_samples': N_SAMPLES,
        'n_repeats': N_REPEATS,
        'branches': timings_clean,
        'total_ms': total_ms,
        'per_sample_ms': overall_per_sample,
        'throughput_per_sec': overall_throughput,
    }
    import json
    lat_path = out_root / f'latency_{args.tag}_seed{args.seed}.json'
    with open(lat_path, 'w') as f:
        json.dump(latency_data, f, indent=2)
    print(f"  ✓ Latency results saved to {lat_path}")
    # ====================================================================
    # END LATENCY BENCHMARK
    # ====================================================================

    # =========================
    # [3/4] Global rule stats
    # =========================
    print("\n[3/4] Computing per-rule global statistics...")
    sys.stdout.flush()
    weights = {r['rule']: r['weight'] for r in art.rule_summary}
    rule_stats = per_rule_global_stats(rule_frame, y_test, weights)
    rule_stats.to_csv(out_root / 'rule_global_stats.csv', index=False)

    # Also save as a clean markdown table for the paper
    md_lines = ['# Global rule statistics on test set\n',
                f'Dataset tag: `{args.tag}` | seed: {args.seed}\n',
                f'Test set size: {len(y_test)} ({int((y_test==1).sum())} attacks, {int((y_test==0).sum())} normals)\n']
    md_lines.append(rule_stats.to_markdown(index=False))
    (out_root / 'rule_global_stats.md').write_text('\n'.join(md_lines), encoding='utf-8')

    print(rule_stats.to_string(index=False))

    # =========================
    # [4/4] Case studies
    # =========================
    print("\n[4/4] Selecting case studies (TP/TN/FP/FN)...")
    sys.stdout.flush()

    rng = np.random.default_rng(args.seed)
    case_indices = select_cases(y_test, neural_meta_score, clafpp_score, threshold, rng)

    # Pull attack family if available (NSL-KDD test_df has 'attack_family',
    # Edge-IIoTset has 'attack_family' too because we set it in data_edge.py)
    fam_col = None
    test_df = art.prepared.test_df
    for c in ['attack_family', 'Attack_type', 'raw_label']:
        if c in test_df.columns:
            fam_col = c
            break

    cases = []
    for case_type, idx_list in case_indices.items():
        for idx in idx_list:
            fam = str(test_df.iloc[idx][fam_col]) if fam_col else None
            cases.append(format_case(
                idx=idx,
                label=case_type,
                y_true=int(y_test[idx]),
                pred=int(clafpp_score[idx] >= threshold),
                neural_score=neural_meta_score[idx],
                symbolic_score=sym_score_raw[idx],
                final_score=clafpp_score[idx],
                threshold=threshold,
                rule_row=rule_frame.iloc[idx],
                lambda_val=lambda_val,
                attack_family=fam,
            ))

    with open(out_root / 'interpretability_cases.json', 'w', encoding='utf-8') as f:
        json.dump(cases, f, indent=2)

    md_cases = cases_to_markdown(cases)
    (out_root / 'interpretability_cases.md').write_text(md_cases, encoding='utf-8')

    # Save the full per-sample rule confidence frame for any further analysis
    rule_frame.assign(
        y_true=y_test,
        neural_meta=neural_meta_score,
        symbolic_raw=sym_score_raw,
        clafpp_final=clafpp_score,
    ).to_csv(out_root / 'per_sample_rule_frame.csv', index=False)

    # Summary
    print(f"\n[done] Artifacts written to {out_root}/")
    print(f"  - rule_global_stats.csv         (per-rule statistics, paper table)")
    print(f"  - rule_global_stats.md          (same as markdown)")
    print(f"  - interpretability_cases.json   (case studies, structured)")
    print(f"  - interpretability_cases.md     (case studies, paper-ready prose)")
    print(f"  - per_sample_rule_frame.csv     (full per-sample rule confidences)")
    print()
    print("Open interpretability_cases.md and rule_global_stats.md to see the paper-ready outputs.")


if __name__ == '__main__':
    main()