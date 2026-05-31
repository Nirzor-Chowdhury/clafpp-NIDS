import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data/raw/edge_iiotset.csv')
y = df['Attack_label'].values

# Drop label-related columns
EXCLUDE = ['Attack_label', 'Attack_type', 'is_vulnerable_trigger']
feature_cols = [c for c in df.columns if c not in EXCLUDE]

print("Single-feature AUROC ranking (top 15):")
print(f"{'Feature':<40} {'AUROC':>8}  {'Notes':<30}")
print('-' * 80)

results = []
for col in feature_cols:
    try:
        x = df[col].astype(float).values
        if np.unique(x).size < 2:
            continue
        auc = roc_auc_score(y, x)
        # Flip if inverted
        auc = max(auc, 1 - auc)
        results.append((col, auc))
    except Exception:
        continue

results.sort(key=lambda r: r[1], reverse=True)
for col, auc in results[:15]:
    flag = "<-- SUSPECT (perfect)" if auc > 0.99 else ("<-- suspicious" if auc > 0.95 else "")
    print(f"{col:<40} {auc:>8.4f}  {flag}")