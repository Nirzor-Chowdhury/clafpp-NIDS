# CLAF++: Conditional Latent Anomaly Fusion with Ensemble Intelligence

Research-grade binary NIDS pipeline for NSL-KDD with:
- Autoencoder + GANomaly for unsupervised anomaly detection
- LSTM for temporal modeling
- Random Forest + XGBoost for tabular strength
- 1D CNN for feature pattern extraction
- Soft Voting, Meta-Learner stacking, and CLAF++ neuro-symbolic fusion

Run a fast smoke example:
```bash
python run_binary_ids.py --config configs/base.yaml --override configs/smoke.yaml
```
