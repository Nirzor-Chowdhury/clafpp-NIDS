from __future__ import annotations
from dataclasses import dataclass
import joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
@dataclass
class TabularProbabilities:
    rf: np.ndarray
    xgb: np.ndarray
    ensemble: np.ndarray
class TabularEnsembleClassifier:
    def __init__(self, config: dict, seed: int=42): self.config=config; self.seed=seed; self.models_={}
    def fit(self,X_train,y_train,X_val,y_val):
        rf_cfg=self.config['models']['rf_xgb']['random_forest']; xgb_cfg=self.config['models']['rf_xgb']['xgboost']; pos=max(int((y_train==1).sum()),1); neg=max(int((y_train==0).sum()),1)
        rf=RandomForestClassifier(n_estimators=int(rf_cfg['n_estimators']), class_weight=rf_cfg.get('class_weight','balanced_subsample'), random_state=self.seed, n_jobs=1); rf.fit(X_train,y_train)
        xgb=XGBClassifier(n_estimators=int(xgb_cfg['n_estimators']), max_depth=int(xgb_cfg['max_depth']), learning_rate=float(xgb_cfg['learning_rate']), subsample=float(xgb_cfg['subsample']), colsample_bytree=float(xgb_cfg['colsample_bytree']), min_child_weight=float(xgb_cfg['min_child_weight']), reg_lambda=float(xgb_cfg['reg_lambda']), objective='binary:logistic', eval_metric='aucpr', scale_pos_weight=float(neg/pos), tree_method='hist', random_state=self.seed, n_jobs=1); xgb.fit(X_train,y_train,verbose=False)
        self.models_={'RandomForest':rf,'XGBoost':xgb}; return self
    def predict_proba(self,X):
        p_rf=self.models_['RandomForest'].predict_proba(X)[:,1].astype(np.float32); p_xgb=self.models_['XGBoost'].predict_proba(X)[:,1].astype(np.float32); return TabularProbabilities(rf=p_rf,xgb=p_xgb,ensemble=(0.45*p_rf+0.55*p_xgb).astype(np.float32))
    def save(self,models_dir): joblib.dump(self.models_['RandomForest'], f'{models_dir}/random_forest.pkl'); joblib.dump(self.models_['XGBoost'], f'{models_dir}/xgboost.pkl')
