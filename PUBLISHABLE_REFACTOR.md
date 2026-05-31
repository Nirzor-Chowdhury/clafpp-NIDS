# CLAF++ Publishable Refactor

## 1. Copy-paste-ready folder architecture

```text
clafpp_ids_pipeline_v5/
├── run_binary_ids.py
├── requirements.txt
├── README.md
├── PUBLISHABLE_REFACTOR.md
├── configs/
│   ├── base.yaml
│   ├── clafpp.yaml
│   └── smoke.yaml
├── artifacts/
│   ├── experiments/
│   │   ├── exp_example_clafpp/
│   │   └── exp_publishable_clafpp/
│   └── reports/
└── src/
    ├── config.py
    ├── data.py
    ├── preprocess.py
    ├── plots.py
    ├── experiment.py
    ├── pipeline.py
    ├── evaluation/
    │   ├── metrics.py
    │   └── evaluator.py
    ├── experiments/
    │   └── ablation.py
    ├── methods/
    │   └── clafpp.py
    ├── models/
    │   ├── autoencoder.py
    │   ├── ganomaly.py
    │   ├── lstm.py
    │   ├── cnn.py
    │   ├── rf_xgb.py
    │   └── meta_learner.py
    ├── symbolic/
    │   ├── rules.py
    │   └── rule_engine.py
    └── training/
        ├── schedule.py
        └── trainer.py
```

## 2. Clean CLAF++ formulation

### Primary formulation

Let the four base experts be:

- `g(x)` = generative anomaly expert from the **AE + GANomaly** branch
- `t(x)` = temporal score from the **LSTM**
- `p(x)` = local feature-pattern score from the **CNN**
- `q(x)` = tabular discriminative score from the **RF/XGBoost** ensemble

All expert outputs are robustly normalized by median-IQR logistic calibration:

\[
\tilde f_i(x) = \sigma\left(\frac{f_i(x)-\operatorname{median}(f_i)}{\operatorname{IQR}(f_i)+\epsilon}\right)
\]

The **primary CLAF++ model** is the stacker:

\[
s_{meta}(x) = M\big([\tilde g(x), \tilde t(x), \tilde p(x), \tilde q(x)]\big)
\]

where `M` is a validation-trained **logistic regression** meta-learner by default.

### Symbolic enhancement layer

The symbolic engine computes a calibrated rule confidence:

\[
r(x) = \sum_{k=1}^{K} \omega_k r_k(x), \qquad \sum_k \omega_k = 1
\]

Each rule weight `ω_k` is learned from validation usefulness rather than hand-set.

The final CLAF++ score is a calibrated residual correction:

\[
S(x) = \operatorname{clip}\left(s_{meta}(x) + \lambda r(x), 0, 1\right)
\]

where `λ` is learned on the validation split.

### Baseline only

Soft voting remains in the codebase only as a baseline:

\[
S_{soft}(x) = \sum_i w_i \tilde f_i(x)
\]

with validation-learned `w_i`.

## 3. Why each model is necessary and non-redundant

### Autoencoder / GANomaly branch
The generative branch is the system’s **out-of-manifold detector**. The autoencoder captures how well a sample can be reconstructed from the normal-traffic manifold, while GANomaly contributes an adversarial consistency signal that reacts to samples that look structurally plausible yet remain semantically off-distribution. Instead of treating AE and GANomaly as two separate headline models, the refactor turns them into **one generative expert**. This reduces over-engineering and gives a clean role: detect abnormality even when labels are sparse or attack families shift.

### LSTM
The LSTM models **temporal dependency** and short-run attack evolution. Intrusions such as flooding, probing, or repeated login abuse often appear not as one isolated packet summary but as a sequence of related states. The LSTM is therefore not a generic extra classifier. It is the expert for **order-sensitive evidence** that tree models and static anomaly scores cannot capture cleanly.

### CNN
The CNN acts as a **local feature-interaction extractor** over the ordered feature vector. In tabular IDS data, correlated neighborhoods of fields often carry suspicious motifs, such as joint deviations in service dispersion, error rates, and host-level statistics. The CNN is useful when the anomaly is expressed through a compact combination of nearby feature groups rather than long temporal history or global reconstruction failure.

### RF / XGBoost
The tree ensemble is the **high-bias-control tabular specialist**. Random Forest provides robust nonparametric averaging, while XGBoost captures sharp decision boundaries and feature interactions efficiently on structured traffic statistics. This branch is especially valuable when the data are already discriminative in tabular space. In CLAF++, the tree ensemble provides the strongest supervised anchor and stabilizes the stacker against the variance of neural experts.

## 4. Refined fusion strategy

- **Soft voting**: baseline only, used for ablation and reviewer-friendly comparison.
- **Meta-learner**: the **primary CLAF++ method**. This is the publishable central contribution because it learns when each expert should be trusted.
- **Neuro-symbolic layer**: enhancement only. It is not a competing fusion head. It is a calibrated residual that improves interpretability and acts as an edge-case safety net.

This removes the earlier ambiguity where three fusion strategies looked like three different main ideas.

## 5. Upgraded symbolic reasoning module

The symbolic engine is now **data-driven**.

Each rule is derived from a learned statistical profile:

1. Build a rule signal from traffic features.
2. Estimate its threshold from the normal-traffic distribution using robust quantiles.
3. Convert threshold exceedance into confidence with a calibrated sigmoid.
4. Learn the rule’s global weight from validation usefulness.

Example rule family:

- `syn_flood_consensus`: high `serror_rate`, `srv_serror_rate`, `dst_host_serror_rate`, and repetitive `count`
- `service_sweep_dispersion`: high `diff_srv_rate`, low `same_srv_rate`, and elevated `srv_diff_host_rate`
- `low_entropy_repetition`: repeated low-diversity behavior using byte-ratio and host-service concentration
- `auth_compromise_pattern`: abnormal `num_failed_logins`, `num_compromised`, `root_shell`, and `is_guest_login`

This is research-grade because the thresholds and weights are **learned from data**, not written as folklore.

## 6. Strong ablation design

The code now supports this clean ablation ladder:

| Ablation | Why it exists |
|---|---|
| AE only | tests pure unsupervised manifold error |
| AE + LSTM | tests whether temporal context improves anomaly evidence |
| AE + CNN | tests whether local feature motifs improve anomaly evidence |
| Ensemble (no meta) | tests whether simple score averaging is enough |
| Meta-Learner (CLAF++) | tests the main research claim: learned stacking beats naive fusion |
| CLAF++ + Symbolic | tests whether the symbolic residual adds value beyond the neural stack |

### Example table from the included smoke artifacts
These numbers are **illustrative smoke-run results**, not final paper claims for full NSL-KDD experiments.

| Model | F1 | AUROC | FAR |
|---|---:|---:|---:|
| AE only | 0.9804 | 0.9989 | 0.0286 |
| Ensemble (no meta) | 1.0000 | 1.0000 | 0.0000 |
| Meta-Learner (CLAF++) | 1.0000 | 1.0000 | 0.0000 |
| CLAF++ + Symbolic | 1.0000 | 1.0000 | 0.0000 |

## 7. Computational efficiency analysis

### Analytic complexity

- **Autoencoder**: `O(N d h)` per epoch for dense encoder-decoder passes
- **GANomaly**: approximately 2 to 3 times AE cost due to encoder-decoder-encoder and discriminator updates
- **LSTM**: `O(N T h^2)` where `T` is sequence length
- **CNN**: `O(N d k c)` for 1D convolutions
- **RF/XGBoost**: tree training cost depends on tree count and split search; inference is fast and parallel-friendly
- **Meta-learner**: negligible compared with the experts, since it only consumes four features
- **Symbolic layer**: linear in the number of rules, effectively negligible at inference

### Real-time IDS argument
The refactor makes real-time deployment plausible because:

- the meta-learner consumes only **four calibrated expert scores**
- the symbolic layer is a **tiny residual** rather than a second heavyweight classifier
- tree inference and symbolic inference are both cheap
- the generative and sequence experts can be batched or run on a separate scoring worker

Operationally, this supports a two-tier IDS setup: cheap supervised and symbolic screening for most traffic, with generative and temporal experts handling high-risk flows.

## 8. Strong but honest research claims

Using the included smoke artifact as an illustration:

- CLAF++ improves AUROC by **0.11 percentage points** over the AE-only baseline.
- CLAF++ reduces false alarm rate by **2.86 percentage points** relative to the AE-only baseline.
- The meta-learner eliminates the ambiguity of manual fusion by learning the relative trust of generative, temporal, pattern, and tabular experts from validation data.
- On the smoke artifact, the symbolic residual preserves top-line metrics while adding **rule-level explanations**, which is still useful because its publishable contribution is interpretability and edge-case handling rather than average-case gain alone.

For the full paper, use these as template claim structures and replace them with full NSL-KDD and repeated-seed results.

## 9. Why CLAF++ works

CLAF++ works because the experts fail differently.

- The **generative branch** is good at novelty and distribution shift.
- The **LSTM** is good at short temporal escalation.
- The **CNN** is good at local interaction motifs.
- The **tree ensemble** is good at crisp tabular boundaries.

A learned stacker reduces both **variance** and **bias** by combining experts under validation supervision instead of naive averaging. The symbolic layer then acts like a glass box on top of a black-box committee: it provides human-readable reasons, catches statistically obvious edge cases, and improves trustworthiness for operational analysts.

## 10. Suggested paper title

**CLAF++: A Stacked Neuro-Symbolic Fusion Framework for Explainable Network Intrusion Detection**

## 11. Suggested abstract

Network intrusion detection remains challenging because no single modeling family captures all aspects of malicious traffic. Unsupervised generative models detect distributional novelty but may underuse label structure, temporal models capture evolving attack dynamics but miss static boundary cues, and classical tabular learners provide strong discriminative performance while offering limited robustness to unseen anomaly patterns. We propose CLAF++, a research-grade intrusion detection framework that combines a generative anomaly expert, temporal modeling, convolutional pattern extraction, and tree-based tabular classification through a validation-trained stacking architecture. In CLAF++, the primary prediction is produced by a lightweight meta-learner over four calibrated expert scores, while a data-driven symbolic residual adds interpretable confidence based on learned statistical rules. This design yields a clean and publishable separation between baseline fusion, primary learned fusion, and explainability enhancement. The framework is fully config-driven, reproducible, and compatible with experiment tracking and ablation analysis. CLAF++ is designed to improve detection robustness while maintaining operational interpretability and deployment feasibility for real-time IDS systems.
