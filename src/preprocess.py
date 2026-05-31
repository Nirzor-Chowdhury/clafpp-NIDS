from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from .config import CATEGORICAL_COLUMNS
@dataclass
class PreprocessArtifacts:
    encoder: ColumnTransformer
    variance_selector: VarianceThreshold
    standard_scaler: StandardScaler
    minmax_scaler: MinMaxScaler
    raw_feature_names: List[str]
    kept_feature_names: List[str]
class Preprocessor:
    def __init__(self, correlation_threshold: float=0.98, variance_threshold: float=0.0):
        self.correlation_threshold=correlation_threshold; self.variance_threshold=variance_threshold; self.corr_keep_indices_=None; self.artifacts_=None
    def _build_encoder(self, X_df: pd.DataFrame):
        # Only include categorical columns that actually exist in the dataframe.
        present_cat = [c for c in CATEGORICAL_COLUMNS if c in X_df.columns]
        if present_cat:
            return ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), present_cat)], remainder='passthrough', verbose_feature_names_out=False)
        # No categorical columns — passthrough everything.
        return ColumnTransformer([], remainder='passthrough', verbose_feature_names_out=False)
    def _fit_correlation_filter(self, X: np.ndarray):
        if X.shape[1]<=1: keep=np.arange(X.shape[1])
        else:
            corr=np.corrcoef(X,rowvar=False); corr=np.nan_to_num(corr); upper=np.triu(np.abs(corr),k=1); drop=set()
            for j in range(upper.shape[1]):
                if np.any(upper[:,j]>self.correlation_threshold): drop.add(j)
            keep=np.array([i for i in range(X.shape[1]) if i not in drop], dtype=int)
        self.corr_keep_indices_=keep.tolist(); return X[:,keep]
    def _apply_correlation_filter(self, X: np.ndarray):
        return X[:, self.corr_keep_indices_]
    def fit(self, X_df: pd.DataFrame):
        enc=self._build_encoder(X_df); X=enc.fit_transform(X_df); raw=list(enc.get_feature_names_out())
        var=VarianceThreshold(threshold=self.variance_threshold); X_var=var.fit_transform(X); kept_after=[raw[i] for i in var.get_support(indices=True)]
        X_f=self._fit_correlation_filter(X_var); kept_final=[kept_after[i] for i in self.corr_keep_indices_]
        std=StandardScaler().fit(X_f); mm=MinMaxScaler().fit(X_f)
        self.artifacts_=PreprocessArtifacts(enc,var,std,mm,raw,kept_final); return self
    def transform_base(self, X_df: pd.DataFrame):
        X=self.artifacts_.encoder.transform(X_df); X=self.artifacts_.variance_selector.transform(X); X=self._apply_correlation_filter(X); return X.astype(np.float32)
    def transform_standard(self, X_df: pd.DataFrame): return self.artifacts_.standard_scaler.transform(self.transform_base(X_df)).astype(np.float32)
    def transform_minmax(self, X_df: pd.DataFrame): return self.artifacts_.minmax_scaler.transform(self.transform_base(X_df)).astype(np.float32)
    def minmax_to_standard(self, X_mm: np.ndarray):
        base=self.artifacts_.minmax_scaler.inverse_transform(X_mm); return self.artifacts_.standard_scaler.transform(base).astype(np.float32)
    @property
    def feature_names_(self): return self.artifacts_.kept_feature_names
