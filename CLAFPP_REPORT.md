# CLAF++ Conference Package

## Folder architecture
```text
clafpp_ids_pipeline_v4/
├── run_binary_ids.py
├── requirements.txt
├── README.md
├── CLAFPP_DELIVERABLE.md
├── configs/
│   ├── base.yaml
│   ├── clafpp.yaml
│   └── smoke.yaml
├── artifacts/
│   ├── reports/
│   │   └── ablation_results.csv
│   └── experiments/
│       └── exp_example_clafpp/
│           ├── config.yaml
│           ├── metrics.json
│           ├── predictions.csv
│           ├── model.pt
│           ├── plots/
│           ├── models/
│           └── reports/
└── src/
    ├── config.py
    ├── data.py
    ├── preprocess.py
    ├── experiment.py
    ├── pipeline.py
    ├── plots.py
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

## CLAF++ formulation
For a sample x,

S(x) = alpha(x) RE(x) + beta(x) T(x) + gamma(x) B(x) + delta(x) P(x) + lambda(x) S_sym(x)

where:
- RE = reconstruction anomaly from Autoencoder plus GANomaly
- T = temporal score from LSTM
- B = tabular score from RF/XGBoost
- P = pattern score from CNN
- S_sym = symbolic rule confidence

All components are robustly normalized on validation-time normal traffic. Base weights are learned from validation AUPRC, and adaptive per-sample weights are computed by a temperature-controlled softmax over normalized component intensities.

## Research contribution
CLAF++ is a hybrid ensemble IDS that combines generative, temporal, tabular, convolutional, and symbolic views of traffic. The novelty is not merely stacking models, but using an adaptive, interpretable anomaly fusion layer that can upweight the most informative expert on a per-sample basis while retaining rule-based explanations for security analysts.

## Experimental results

Results from the real NSL-KDD experiment (`artifacts/experiments/exp_051/`)
will be tabulated here after the Day 1 re-run with bug fixes applied.
