from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..data import get_feature_columns, load_nsl_kdd, make_smoke_test_frames, split_train_validation
from ..methods import LearnedWeightedFusion, ResidualSymbolicEnhancer, SoftVotingFusion
from ..models import (
    AutoencoderDetector,
    CNNPatternDetector,
    GANomalyDetector,
    SequenceLSTMDetector,
    StackedMetaLearner,
    TabularEnsembleClassifier,
)
from ..preprocess import Preprocessor
from ..symbolic import NeuroSymbolicRuleEngine
from .schedule import TrainingSchedule


@dataclass
class PreparedData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]
    X_train_std: np.ndarray
    X_val_std: np.ndarray
    X_test_std: np.ndarray
    X_train_mm: np.ndarray
    X_val_mm: np.ndarray
    X_test_mm: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    preprocessor: Preprocessor


@dataclass
class CLAFPPArtifacts:
    prepared: PreparedData
    autoencoder: object
    ganomaly: object
    lstm: object
    cnn: object
    tabular: object
    meta_learner: object
    generative_fusion: object
    soft_voting: object
    symbolic_engine: object
    symbolic_enhancer: object
    symbolic_val: object
    symbolic_test: object
    training_times: dict[str, float]
    predictions: dict[str, dict[str, np.ndarray]]
    rule_summary: list[dict]
    feature_names: list[str]
    meta_feature_names: list[str]
    schedule: TrainingSchedule


class ResearchTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config['runtime']['seed'])
        self.schedule = TrainingSchedule(config)

    def _load_frames(self):
        runtime = self.config['runtime']
        dataset = self.config['dataset']
        if runtime.get('smoke_test', False):
            return make_smoke_test_frames(
                n_train=int(dataset.get('smoke_train_rows', 300)),
                n_test=int(dataset.get('smoke_test_rows', 120)),
                seed=self.seed,
            )
        # NEW: dispatch by dataset version
        if dataset.get('version', 'NSL-KDD') == 'Edge-IIoTset':
            from ..data_edge import load_edge_iiotset
            csv = Path(runtime['raw_data_dir']) / dataset['csv_file']
            if not csv.exists():
                raise FileNotFoundError(f'Missing Edge-IIoTset CSV at {csv}')
            return load_edge_iiotset(csv, test_size=float(dataset.get('test_size', 0.25)), seed=self.seed)
        # default: NSL-KDD path
        raw = Path(runtime['raw_data_dir'])
        train = raw / dataset['train_file']
        test = raw / dataset['test_file']
        if not train.exists() or not test.exists():
            raise FileNotFoundError(f'Missing NSL-KDD files. Expected {train} and {test}.')
        return load_nsl_kdd(train, test)

    def prepare_data(self):
        train_df, test_df = self._load_frames()
        train_df, val_df = split_train_validation(train_df, float(self.config['dataset']['val_size']), self.seed)
        
        if self.config['dataset'].get('version', 'NSL-KDD') == 'Edge-IIoTset':
            from ..data_edge import get_edge_feature_columns
            feat = get_edge_feature_columns(train_df)
        else:
            feat = get_feature_columns(train_df)
            
        Xtr = train_df[feat].copy()
        Xv = val_df[feat].copy()
        Xte = test_df[feat].copy()
        ytr = train_df['target'].to_numpy().astype(np.int32)
        yv = val_df['target'].to_numpy().astype(np.int32)
        yte = test_df['target'].to_numpy().astype(np.int32)
        pre = Preprocessor(
            correlation_threshold=float(self.config['preprocessing']['correlation_threshold']),
            variance_threshold=float(self.config['preprocessing']['variance_threshold']),
        ).fit(Xtr)
        return PreparedData(
            train_df,
            val_df,
            test_df,
            feat,
            pre.transform_standard(Xtr),
            pre.transform_standard(Xv),
            pre.transform_standard(Xte),
            pre.transform_minmax(Xtr),
            pre.transform_minmax(Xv),
            pre.transform_minmax(Xte),
            ytr,
            yv,
            yte,
            pre,
        )

    @staticmethod
    def _minmax01(x):
        x = np.asarray(x, dtype=np.float32)
        lo = float(x.min())
        hi = float(x.max())
        return np.zeros_like(x, dtype=np.float32) if hi - lo < 1e-8 else ((x - lo) / (hi - lo)).astype(np.float32)

    def run(self):
        cfg = self.config
        p = self.prepare_data()
        tt: dict[str, float] = {}

        t = time.perf_counter()
        ae = AutoencoderDetector(input_dim=p.X_train_mm.shape[1], **cfg['models']['autoencoder']).fit(p.X_train_mm[p.y_train == 0])
        tt['autoencoder_seconds'] = float(time.perf_counter() - t)

        t = time.perf_counter()
        gan = GANomalyDetector(input_dim=p.X_train_mm.shape[1], **cfg['models']['ganomaly']).fit(p.X_train_mm[p.y_train == 0])
        tt['ganomaly_seconds'] = float(time.perf_counter() - t)

        t = time.perf_counter()
        lstm = SequenceLSTMDetector(input_dim=p.X_train_std.shape[1], **cfg['models']['lstm']).fit(p.X_train_std, p.y_train)
        tt['lstm_seconds'] = float(time.perf_counter() - t)

        t = time.perf_counter()
        tab = TabularEnsembleClassifier(cfg, seed=self.seed).fit(p.X_train_std, p.y_train, p.X_val_std, p.y_val)
        tt['tabular_seconds'] = float(time.perf_counter() - t)

        t = time.perf_counter()
        cnn = CNNPatternDetector(input_dim=p.X_train_std.shape[1], **cfg['models']['cnn']).fit(p.X_train_std, p.y_train)
        tt['cnn_seconds'] = float(time.perf_counter() - t)

        ae_tr, ae_v, ae_te = ae.score_samples(p.X_train_mm), ae.score_samples(p.X_val_mm), ae.score_samples(p.X_test_mm)
        gan_tr, gan_v, gan_te = gan.score_samples(p.X_train_mm), gan.score_samples(p.X_val_mm), gan.score_samples(p.X_test_mm)
        lstm_tr, lstm_v, lstm_te = lstm.predict_proba(p.X_train_std), lstm.predict_proba(p.X_val_std), lstm.predict_proba(p.X_test_std)
        cnn_tr, cnn_v, cnn_te = cnn.predict_proba(p.X_train_std), cnn.predict_proba(p.X_val_std), cnn.predict_proba(p.X_test_std)
        tab_tr, tab_v, tab_te = tab.predict_proba(p.X_train_std), tab.predict_proba(p.X_val_std), tab.predict_proba(p.X_test_std)

        ae_tr_raw = ae_tr['reconstruction_error'] + 0.35 * ae_tr['latent_distance']
        ae_v_raw = ae_v['reconstruction_error'] + 0.35 * ae_v['latent_distance']
        ae_te_raw = ae_te['reconstruction_error'] + 0.35 * ae_te['latent_distance']

        gan_tr_raw = gan_tr['ganomaly_score']
        gan_v_raw = gan_v['ganomaly_score']
        gan_te_raw = gan_te['ganomaly_score']

        generative_fusion = LearnedWeightedFusion(weight_floor=float(cfg['fusion'].get('generative_branch', {}).get('weight_floor', 0.08)))
        generative_fusion.fit(p.y_val, {'autoencoder': ae_v_raw, 'ganomaly': gan_v_raw})
        gen_tr, _ = generative_fusion.score({'autoencoder': ae_tr_raw, 'ganomaly': gan_tr_raw})
        gen_v, gen_val_details = generative_fusion.score({'autoencoder': ae_v_raw, 'ganomaly': gan_v_raw})
        gen_te, _ = generative_fusion.score({'autoencoder': ae_te_raw, 'ganomaly': gan_te_raw})

        temporal_tr, temporal_v, temporal_te = self._minmax01(lstm_tr), self._minmax01(lstm_v), self._minmax01(lstm_te)
        pattern_tr, pattern_v, pattern_te = self._minmax01(cnn_tr), self._minmax01(cnn_v), self._minmax01(cnn_te)
        tabular_tr, tabular_v, tabular_te = self._minmax01(tab_tr.ensemble), self._minmax01(tab_v.ensemble), self._minmax01(tab_te.ensemble)

        soft = SoftVotingFusion(weight_floor=float(cfg['fusion']['soft_voting'].get('weight_floor', 0.10)))
        soft.fit(p.y_val, {'generative': gen_v, 'temporal': temporal_v, 'tabular': tabular_v, 'pattern': pattern_v})
        soft_v, _ = soft.score({'generative': gen_v, 'temporal': temporal_v, 'tabular': tabular_v, 'pattern': pattern_v})
        soft_te, _ = soft.score({'generative': gen_te, 'temporal': temporal_te, 'tabular': tabular_te, 'pattern': pattern_te})

        meta_names = ['generative_score', 'temporal_score', 'pattern_score', 'tabular_score']

        # =========================
        # BASE META FEATURES
        # =========================
        X_meta_v = np.column_stack([gen_v, temporal_v, pattern_v, tabular_v]).astype(np.float32)
        X_meta_te = np.column_stack([gen_te, temporal_te, pattern_te, tabular_te]).astype(np.float32)

        # =========================
        # FEATURE ENGINEERING
        # =========================
        for arr in (X_meta_v, X_meta_te):
            pass  # marker comment: kept as-is below for clarity

        max_v = np.max(X_meta_v, axis=1, keepdims=True)
        min_v = np.min(X_meta_v, axis=1, keepdims=True)
        mean_v = np.mean(X_meta_v, axis=1, keepdims=True)
        std_v = np.std(X_meta_v, axis=1, keepdims=True)
        dis_v = max_v - min_v

        max_te = np.max(X_meta_te, axis=1, keepdims=True)
        min_te = np.min(X_meta_te, axis=1, keepdims=True)
        mean_te = np.mean(X_meta_te, axis=1, keepdims=True)
        std_te = np.std(X_meta_te, axis=1, keepdims=True)
        dis_te = max_te - min_te

        X_meta_v = np.hstack([X_meta_v, max_v, min_v, mean_v, std_v, dis_v])
        X_meta_te = np.hstack([X_meta_te, max_te, min_te, mean_te, std_te, dis_te])

        meta_names += ['max_score', 'min_score', 'mean_score', 'std_score', 'disagreement']

        # =========================
        # 🔥 SPLIT VAL: prevent leakage between meta-fitting and calibration
        # =========================
        from sklearn.model_selection import train_test_split as _tts
        meta_train_idx, calib_idx = _tts(
            np.arange(len(p.y_val)),
            test_size=0.40,
            stratify=p.y_val,
            random_state=self.seed,
        )
        X_meta_train_split = X_meta_v[meta_train_idx]
        y_meta_train_split = p.y_val[meta_train_idx]
        X_meta_calib = X_meta_v[calib_idx]
        y_meta_calib = p.y_val[calib_idx]

        # Normalization fit ONLY on meta-training portion to avoid leakage into calib.
        mean_meta = X_meta_train_split.mean(axis=0)
        std_meta = X_meta_train_split.std(axis=0) + 1e-6
        X_meta_train_split = (X_meta_train_split - mean_meta) / std_meta
        X_meta_calib = (X_meta_calib - mean_meta) / std_meta
        X_meta_te = (X_meta_te - mean_meta) / std_meta

        # =========================
        # META LEARNER (trained on meta_train portion only)
        # =========================
        meta_cfg = cfg['fusion']['meta_learner']

        t = time.perf_counter()
        meta = StackedMetaLearner(
            kind=meta_cfg.get('type', 'logistic_regression'),
            seed=self.seed,
            params=meta_cfg.get('params', {})
        ).fit(X_meta_train_split, y_meta_train_split, feature_names=meta_names)
        tt['meta_learner_seconds'] = float(time.perf_counter() - t)

        # Predict on calib + test (note: meta_v reflects the FULL val set, so
        # downstream code that reports val predictions stays intact)
        X_meta_v_full = np.vstack([X_meta_train_split, X_meta_calib])
        # Rebuild full-val ordering by index
        meta_v_full = np.empty(len(p.y_val), dtype=np.float32)
        meta_v_full[meta_train_idx] = meta.predict_proba(X_meta_train_split)
        meta_v_full[calib_idx] = meta.predict_proba(X_meta_calib)
        meta_v = meta_v_full
        meta_te = meta.predict_proba(X_meta_te)

        # =========================
        # SYMBOLIC ENGINE  (unchanged)
        # =========================
        if cfg['dataset'].get('version', 'NSL-KDD') == 'Edge-IIoTset':
            from ..symbolic.rules_edge import edge_rule_definitions
            rule_defs = edge_rule_definitions()
        else:
            rule_defs = None  # uses default NSL-KDD rules
        symbolic_engine = NeuroSymbolicRuleEngine(
            rules=rule_defs,
            weight_floor=float(cfg['fusion'].get('symbolic', {}).get('weight_floor', 0.05))
        )
        symbolic_engine.fit(
            p.train_df.reset_index(drop=True), p.y_train,
            p.val_df.reset_index(drop=True), p.y_val
        )
        sym_v_out = symbolic_engine.evaluate(p.val_df.reset_index(drop=True))
        sym_te_out = symbolic_engine.evaluate(p.test_df.reset_index(drop=True))
        sym_v = sym_v_out.score
        sym_te = sym_te_out.score

        # =========================
        # SYMBOLIC ENHANCER (fit on CALIB portion only — meta never saw it)
        # =========================
        t = time.perf_counter()
        symbolic_enhancer = ResidualSymbolicEnhancer(
            lambda_grid=np.linspace(
                0.0,
                float(cfg['fusion'].get('symbolic', {}).get('max_lambda', 1.0)),
                int(cfg['fusion'].get('symbolic', {}).get('lambda_steps', 41))
            ),
            objective=str(cfg['fusion'].get('symbolic', {}).get('objective', 'auprc')),
        )
        symbolic_enhancer.fit(
            y_meta_calib,
            meta_v[calib_idx],
            sym_v[calib_idx],
        )
        tt['symbolic_seconds'] = float(time.perf_counter() - t)

        # final outputs
        claf_v, _ = symbolic_enhancer.score(meta_v, sym_v)
        claf_te, claf_details = symbolic_enhancer.score(meta_te, sym_te)

        tt['soft_voting_seconds'] = 0.0
        tt['total_seconds'] = float(sum(tt.values()))

        preds_val = {
            'ae': self._minmax01(ae_v_raw),
            'ganomaly': self._minmax01(gan_v_raw),
            'generative': gen_v,
            'lstm': temporal_v,
            'cnn': pattern_v,
            'rf': tab_v.rf.astype(np.float32),
            'xgb': tab_v.xgb.astype(np.float32),
            'tabular': tabular_v,
            'soft_voting': soft_v,
            'meta_learner': meta_v,
            'symbolic': sym_v,
            'clafpp': claf_v,
            'rule_explanations': np.array(sym_v_out.explanations, dtype=object),
        }
        preds_test = {
            'ae': self._minmax01(ae_te_raw),
            'ganomaly': self._minmax01(gan_te_raw),
            'generative': gen_te,
            'lstm': temporal_te,
            'cnn': pattern_te,
            'rf': tab_te.rf.astype(np.float32),
            'xgb': tab_te.xgb.astype(np.float32),
            'tabular': tabular_te,
            'soft_voting': soft_te,
            'meta_learner': meta_te,
            'symbolic': sym_te,
            'clafpp': claf_te,
            'rule_explanations': np.array(sym_te_out.explanations, dtype=object),
        }

        return CLAFPPArtifacts(
            prepared=p,
            autoencoder=ae,
            ganomaly=gan,
            lstm=lstm,
            cnn=cnn,
            tabular=tab,
            meta_learner=meta,
            generative_fusion=generative_fusion,
            soft_voting=soft,
            symbolic_engine=symbolic_engine,
            symbolic_enhancer=symbolic_enhancer,
            symbolic_val=sym_v,
            symbolic_test=sym_te,
            training_times=tt,
            predictions={
                'val': preds_val,
                'test': preds_test,
                'generative_validation_details': gen_val_details,
                'clafpp_details': claf_details,
            },
            rule_summary=sym_te_out.summary,
            feature_names=p.preprocessor.feature_names_,
            meta_feature_names=meta_names,
            schedule=self.schedule,
        )
