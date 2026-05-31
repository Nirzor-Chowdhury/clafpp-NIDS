# diagnose_leak_2.py
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv('data/raw/edge_iiotset.csv')
y = df['Attack_label'].values
EXCLUDE = ['Attack_label', 'Attack_type', 'is_vulnerable_trigger', 'dns.qry.name.len']
feature_cols = [c for c in df.columns if c not in EXCLUDE]
X = df[feature_cols].astype(float).values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
preds = rf.predict(X_te)
f1 = f1_score(y_te, preds)
print(f"\nFull RF F1 (with dns.qry.name.len removed): {f1:.4f}")
if f1 > 0.99:
    print("→ STILL PERFECT — more leakers present.\n")

# Feature importance ranking
imp = sorted(zip(feature_cols, rf.feature_importances_), key=lambda t: t[1], reverse=True)
print("Top 20 features by RF importance:")
print(f"{'Feature':<40} {'Importance':>12}")
print('-' * 56)
for col, val in imp[:20]:
    flag = "<-- likely leaker" if val > 0.05 else ""
    print(f"{col:<40} {val:>12.4f}  {flag}")

# Now test: does dropping the top-3 features bring F1 below 0.99?
print("\nProgressive ablation — drop features one at a time until F1 < 0.99:")
to_drop = []
for col, _ in imp:
    to_drop.append(col)
    remaining = [c for c in feature_cols if c not in to_drop]
    if len(remaining) < 5: break
    X_tr_a = df.loc[X_tr.shape[0] * [True] + (len(df) - X_tr.shape[0]) * [False], remaining]
    # Simpler: just refit on the same split
    keep_idx = [feature_cols.index(c) for c in remaining]
    rf2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf2.fit(X_tr[:, keep_idx], y_tr)
    f1_2 = f1_score(y_te, rf2.predict(X_te[:, keep_idx]))
    print(f"  After dropping {col:<35}  F1 = {f1_2:.4f}")
    if f1_2 < 0.99:
        print(f"\n→ Dropping {to_drop} brings RF below perfect.")
        print(f"  Honest baseline F1: {f1_2:.4f}")
        break