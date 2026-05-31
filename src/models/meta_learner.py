from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, recall_score, fbeta_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

class StackedMetaLearner:
    """Enhanced fusion model for CLAF++ (FIXED VERSION)"""

    def __init__(self, kind: str = 'xgboost', seed: int = 42, params: dict | None = None):
        self.kind = kind
        self.seed = seed
        self.params = params or {}
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.threshold_ = 0.5
        self.feature_names_ = None

    # 🔥 Feature Engineering (IMPORTANT)
    def _augment_features(self, X):
        mean_feat = np.mean(X, axis=1, keepdims=True)
        max_feat = np.max(X, axis=1, keepdims=True)
        std_feat = np.std(X, axis=1, keepdims=True)
        return np.hstack([X, mean_feat, max_feat, std_feat])

    def fit(self, X_meta_val: np.ndarray, y_val: np.ndarray, feature_names=None):

        # 🔥 Step 1: Feature Engineering
        X_meta_val = self._augment_features(X_meta_val)

        # 🔥 Step 2: Scaling
        X_meta_val = self.scaler_.fit_transform(X_meta_val)

        self.feature_names_ = feature_names or [f'f_{i}' for i in range(X_meta_val.shape[1])]

        # 🔥 Step 3: Model Selection
        if self.kind == 'mlp':
            base_model = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                random_state=self.seed,
                max_iter=500
            )

        elif self.kind == 'xgboost':
            base_model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=3,   # 🔥 FIX: improve recall
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=self.seed
            )

        else:
            base_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.seed,
            )

        # 🔥 Step 4: Calibration (VERY IMPORTANT)
        self.model_ = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
        self.model_.fit(X_meta_val, y_val)

        # 🔥 Step 5: Threshold Optimization (CRITICAL FIX)
        probs = self.model_.predict_proba(X_meta_val)[:, 1]

        best_score = 0
        best_thresh = 0.5

        for t in np.linspace(0.05, 0.6, 110):
            preds = (probs >= t).astype(int)

            recall = recall_score(y_val, preds)

            tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
            false_alarm_rate = fp / (fp + tn + 1e-6)

            # F-beta with beta=2 — favors recall, which is what IDS deployments care about.
            # A beta of 2 means recall is weighted 4x precision in the F-measure.
            score = fbeta_score(y_val, preds, beta=2)

            if score > best_score:
                best_score = score
                best_thresh = t

        self.threshold_ = best_thresh
        print(f"[Meta] Best threshold: {self.threshold_:.3f} | Score: {best_score:.4f}")

        return self

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        X_meta = self._augment_features(X_meta)
        X_meta = self.scaler_.transform(X_meta)
        return self.model_.predict_proba(X_meta)[:, 1].astype(np.float32)

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_meta)
        return (probs >= self.threshold_).astype(int)

    def interpret(self) -> dict:
        return {
            'type': self.kind,
            'threshold': float(self.threshold_)
        }

    def save(self, path: str):
        joblib.dump({
            'model': self.model_,
            'scaler': self.scaler_,
            'threshold': self.threshold_,
            'feature_names': self.feature_names_
        }, path)