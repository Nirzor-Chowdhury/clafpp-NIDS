#!/usr/bin/env python3
"""
CLAFPP++ publication figures.
Generates three figures (PDF + PNG, 300 dpi) for the JNSM submission:
  A) Reliability / calibration curves (two datasets) + ECE bars
  B) Edge-IIoTset leakage audit histograms (6 features)
  C) Per-prediction symbolic attribution (NSL-KDD), safe lambda*=0 form
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
import pandas as pd
import os

PROJECT = r"D:\NIDS\New NIds\CLAF ++\clafpp_ids_pipeline_v5\clafpp_ids_pipeline_v5"
OUT = os.path.join(PROJECT, "figures")
os.makedirs(OUT, exist_ok=True)
DIAG = os.path.join(PROJECT, "artifacts", "interpretability", "{0}", "clafpp_diagnostics_{0}_seed42.npz")
RAW_EDGE = os.path.join(PROJECT, "data", "raw", "edge_iiotset.csv")

# ── Global style: clean, academic, colorblind-friendly ──────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'figure.dpi': 120,
})
# Okabe-Ito colorblind-safe palette
C_CLAFPP   = '#0072B2'  # blue
C_ENSEMBLE = '#D55E00'  # vermillion
C_RFXGB    = '#009E73'  # green
C_XGB      = '#CC79A7'  # purple
C_NORMAL   = '#56B4E9'  # light blue
C_ATTACK   = '#E69F00'  # orange


def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    e, N = 0.0, len(y_true)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / N) * abs(y_true[m].mean() - y_prob[m].mean())
    return e


# ════════════════════════════════════════════════════════════════════════
# FIGURE A — CALIBRATION
# ════════════════════════════════════════════════════════════════════════
def figure_calibration():
    """Two-panel reliability curves with non-overlapping ECE boxes (Option 2)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7))

    # (tag, title, n_bins) — fewer bins on NSL-KDD smooths the bimodal curve
    datasets = [('nslkdd', 'NSL-KDD (KDDTest+)', 6),
                ('edge', 'Edge-IIoTset', 10)]
    models = [
        ('p_meta',            'CLAFPP++ (Meta)',    C_CLAFPP,   'o', '-'),
        ('p_ensemble_nometa', 'Ensemble (no meta)', C_ENSEMBLE, 's', '--'),
        ('p_rfxgb',           'RF/XGB',             C_RFXGB,    '^', '-.'),
        ('p_xgb',             'XGBoost',            C_XGB,      'D', ':'),
    ]

    for col, (tag, title, n_bins) in enumerate(datasets):
        d = np.load(DIAG.format(tag), allow_pickle=True)
        y = d['y_test']
        ax = axes[col]
        ax.plot([0, 1], [0, 1], 'k:', lw=1.0, label='Perfect calibration', zorder=1)

        eces = {}
        for key, name, color, marker, ls in models:
            p = d[key]
            eces[name] = ece(y, p)
            frac_pos, mean_pred = calibration_curve(
                y, p, n_bins=n_bins, strategy='quantile')
            ax.plot(mean_pred, frac_pos, marker=marker, ls=ls, color=color,
                    ms=4, lw=1.4, label=name, zorder=3, alpha=0.9)

        ax.set_xlabel('Mean predicted probability')
        if col == 0:
            ax.set_ylabel('Observed fraction of attacks')
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25, lw=0.5); ax.set_aspect('equal')

        # Shared legend only on the right (Edge) panel, lower-right
        if col == 1:
            ax.legend(loc='lower right', framealpha=0.95, edgecolor='0.8', fontsize=7)

        ece_txt = '\n'.join([
            "ECE (down):",
            f"  CLAFPP++: {eces['CLAFPP++ (Meta)']:.3f}",
            f"  Ensemble: {eces['Ensemble (no meta)']:.3f}",
            f"  RF/XGB:   {eces['RF/XGB']:.3f}",
            f"  XGBoost:  {eces['XGBoost']:.3f}",
        ])
        if col == 0:
            # NSL-KDD: no legend here -> lower-right is clear
            ax.text(0.97, 0.05, ece_txt, transform=ax.transAxes, fontsize=6.8,
                    ha='right', va='bottom', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.7', alpha=0.95))
        else:
            # Edge: legend at lower-right -> ECE in empty upper-left
            ax.text(0.03, 0.97, ece_txt, transform=ax.transAxes, fontsize=6.8,
                    ha='left', va='top', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.7', alpha=0.95))

    fig.suptitle('Reliability diagrams: CLAFPP++ meta-learner vs. uncalibrated baselines (seed 42)',
                 fontsize=10.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{OUT}/fig_calibration.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig_calibration.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("\u2713 Figure A (calibration) saved")
    for tag, title, _ in datasets:
        d = np.load(DIAG.format(tag), allow_pickle=True)
        y = d['y_test']
        print(f"  {title}: CLAFPP++ ECE={ece(y, d['p_meta']):.4f}  "
              f"Ensemble ECE={ece(y, d['p_ensemble_nometa']):.4f}")



# ════════════════════════════════════════════════════════════════════════
# FIGURE B — LEAKAGE AUDIT
# ════════════════════════════════════════════════════════════════════════
def figure_leakage():
    df = pd.read_csv(RAW_EDGE, low_memory=False)
    feats = ['dns.qry.name.len', 'mqtt.topic', 'mqtt.conack.flags',
             'mqtt.protoname', 'mqtt.msg', 'tcp.dstport']
    y = df['Attack_label'].values
    from sklearn.metrics import roc_auc_score

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
    axes = axes.ravel()

    for i, feat in enumerate(feats):
        ax = axes[i]
        x = df[feat].values.astype(float)
        try:
            a = roc_auc_score(y, x)
            auroc = max(a, 1 - a)
        except Exception:
            auroc = float('nan')

        n_unique = len(np.unique(x))

        if feat == 'tcp.dstport':
            # Continuous, weakest leaker — show as class-conditional histogram
            # (proportion within class) on a clipped range.
            bins = np.linspace(0, 65535, 30)
            wn = np.ones((y == 0).sum()) / (y == 0).sum()
            wa = np.ones((y == 1).sum()) / (y == 1).sum()
            ax.hist(x[y == 0], bins=bins, weights=wn, alpha=0.6, color=C_NORMAL,
                    label='Normal', edgecolor='white', linewidth=0.3)
            ax.hist(x[y == 1], bins=bins, weights=wa, alpha=0.6, color=C_ATTACK,
                    label='Attack', edgecolor='white', linewidth=0.3)
            ax.set_xlabel('Port number', fontsize=7.5)
            ax.set_ylabel('Proportion within class', fontsize=7)
        else:
            # Discrete encoded feature — grouped bars of class-conditional
            # proportion at each value. Collapse rare tail values into "other".
            if feat == 'mqtt.msg':
                display_vals = [0, 1]
                tail_label = '≥2'
            else:
                display_vals = sorted([v for v in np.unique(x) if v <= 2])
                tail_label = f'≥{int(max(display_vals))+1}'
            normal_props, attack_props = [], []
            for v in display_vals:
                normal_props.append(np.mean(x[y == 0] == v))
                attack_props.append(np.mean(x[y == 1] == v))
            # tail
            tail_n = np.mean(x[y == 0] > max(display_vals))
            tail_a = np.mean(x[y == 1] > max(display_vals))
            if tail_n > 0.001 or tail_a > 0.001:
                normal_props.append(tail_n)
                attack_props.append(tail_a)
                xt_labels = [str(int(v)) for v in display_vals] + [tail_label]
            else:
                xt_labels = [str(int(v)) for v in display_vals]
            xpos = np.arange(len(xt_labels))
            w = 0.38
            ax.bar(xpos - w/2, normal_props, w, color=C_NORMAL, label='Normal',
                   edgecolor='white', linewidth=0.4)
            ax.bar(xpos + w/2, attack_props, w, color=C_ATTACK, label='Attack',
                   edgecolor='white', linewidth=0.4)
            ax.set_xticks(xpos)
            ax.set_xticklabels(xt_labels)
            ax.set_xlabel('Encoded value', fontsize=7.5)
            ax.set_ylabel('Proportion within class', fontsize=7)
            ax.set_ylim(0, 1.08)

        ax.set_title(f'{feat}\n(AUROC = {auroc:.4f})', fontsize=8)
        ax.grid(axis='y', alpha=0.2, lw=0.4)
        if i == 0:
            ax.legend(loc='upper center', fontsize=7, framealpha=0.95)

    fig.suptitle('Edge-IIoTset leakage audit: class-conditional distributions of removed features',
                 fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{OUT}/fig_leakage.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig_leakage.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("✓ Figure B (leakage) saved")


# ════════════════════════════════════════════════════════════════════════
# FIGURE C — SYMBOLIC ATTRIBUTION (safe lambda*=0 form)
# ════════════════════════════════════════════════════════════════════════
def figure_attribution():
    d = np.load(DIAG.format('nslkdd'), allow_pickle=True)
    y = d['y_test']
    p_meta = d['p_meta']
    p_clafpp = d['p_clafpp']
    acts = d['rule_activations']           # (n, 4)
    rule_names = [str(r) for r in d['rule_names']]
    rule_weights = d['rule_weights']
    lam = float(d['lambda_star'])
    thr = float(d['threshold'])

    # Pick a clear TP (high meta score, attack) and FN (low meta score, attack)
    pred = (p_clafpp >= thr).astype(int)
    tp_mask = (pred == 1) & (y == 1)
    fn_mask = (pred == 0) & (y == 1)
    # TP: prefer a sample where two or more rules fire (richer attribution)
    tp_two = np.where(tp_mask & ((acts >= 0.55).sum(axis=1) >= 2))[0]
    if len(tp_two):
        tp_idx = tp_two[np.argmax(p_meta[tp_two])]
    else:
        tp_one = np.where(tp_mask & (acts.max(axis=1) >= 0.55))[0]
        tp_idx = tp_one[np.argmax(p_meta[tp_one])] if len(tp_one) else np.where(tp_mask)[0][0]
    fn_candidates = np.where(fn_mask)[0]
    fn_idx = fn_candidates[np.argmin(p_meta[fn_candidates])] if len(fn_candidates) else np.where(fn_mask)[0][0]

    short = {
        'service_sweep_dispersion': 'service_sweep',
        'syn_flood_consensus': 'syn_flood',
        'auth_compromise_pattern': 'auth_compromise',
        'low_entropy_repetition': 'low_entropy_rep',
    }
    labels = [short.get(r, r) for r in rule_names]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    for ax, idx, case in [(axes[0], tp_idx, 'True Positive'),
                          (axes[1], fn_idx, 'False Negative')]:
        rule_a = acts[idx]
        order = np.argsort(rule_a)[::-1]
        ypos = np.arange(len(rule_names))
        colors = [C_CLAFPP if rule_a[o] >= 0.55 else '0.7' for o in order]
        ax.barh(ypos, rule_a[order], color=colors, height=0.6,
                edgecolor='white', linewidth=0.5)
        ax.axvline(0.55, color=C_ATTACK, ls='--', lw=1.0, label='fire threshold (0.55)')
        ax.set_yticks(ypos)
        ax.set_yticklabels([labels[o] for o in order], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Rule activation', fontsize=8)
        ax.grid(axis='x', alpha=0.2, lw=0.4)
        fam = ''
        ax.set_title(f'{case}  (test #{idx})\n'
                     f'neural meta = {p_meta[idx]:.4f}, '
                     f'final = {p_clafpp[idx]:.4f}', fontsize=8.5)
        ax.legend(loc='lower right', fontsize=7)

    fig.suptitle(
        rf'Per-prediction symbolic attribution on NSL-KDD (seed 42, $\lambda^\star = {lam:.2f}$): '
        'rule activations are an interpretive overlay',
        fontsize=9.0, y=1.04)
    # annotation explaining lambda=0
    fig.text(0.5, -0.06,
             r'With $\lambda^\star=0$ the final score equals the neural meta score; '
             'rule activations explain which attack signatures are present without '
             'altering the decision.',
             ha='center', fontsize=7.5, style='italic')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT}/fig_attribution.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig_attribution.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✓ Figure C (attribution) saved  [TP #{tp_idx}, FN #{fn_idx}]")


if __name__ == '__main__':
    figure_calibration()
    figure_leakage()
    figure_attribution()
    print("\nAll figures written to", OUT)
