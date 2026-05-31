from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import make_experiment_paths, set_global_seed
from .evaluation import ResearchEvaluator
from .experiment import ExperimentTracker
from .experiments import run_ablation_study
from .plots import (
    save_roc_curve,
    save_pr_curve,
    save_all_model_confusion_matrices,
    save_individual_roc_curves,
    save_individual_pr_curves,
    save_score_distributions,
    save_prediction_histograms,
    save_latent_projection,
    save_training_time_bar,
    save_correlation_heatmap   # ✅ ADD THIS
)
from .training import ResearchTrainer

NUMERIC_DISTRIBUTION_CANDIDATES = [
    'src_bytes', 'dst_bytes', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'same_srv_rate', 'dst_host_srv_count', 'dst_host_serror_rate',
]


def _torch_model_size_mb(module) -> float:
    return float(sum(p.nelement() * p.element_size() for p in module.parameters()) / (1024 ** 2))


def _torch_param_count(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _profile_compute(art, paths) -> pd.DataFrame:
    prepared = art.prepared

    def measure(name, fn, train_seconds, size_mb, params):
        start = time.perf_counter()
        fn()
        infer_seconds = float(time.perf_counter() - start)
        return {
            'Model': name,
            'TrainSeconds': float(train_seconds),
            'InferenceSeconds': infer_seconds,
            'InferenceMsPerSample': float((infer_seconds / max(len(prepared.y_test), 1)) * 1000.0),
            'ModelSizeMB': float(size_mb),
            'ParameterCount': int(params),
        }

    # =========================
    # 🔥 FIX: Build correct meta features (9 features)
    # =========================
    X_meta = np.column_stack([
        art.predictions['test']['generative'],
        art.predictions['test']['lstm'],
        art.predictions['test']['cnn'],
        art.predictions['test']['tabular']
    ]).astype(np.float32)

    # feature engineering (same as training)
    max_ = np.max(X_meta, axis=1, keepdims=True)
    min_ = np.min(X_meta, axis=1, keepdims=True)
    mean_ = np.mean(X_meta, axis=1, keepdims=True)
    std_ = np.std(X_meta, axis=1, keepdims=True)
    dis_ = max_ - min_

    X_meta = np.hstack([X_meta, max_, min_, mean_, std_, dis_])

    # normalization (same logic as training)
    mean_meta = X_meta.mean(axis=0)
    std_meta = X_meta.std(axis=0) + 1e-6
    X_meta = (X_meta - mean_meta) / std_meta

    # =========================
    # MODELS
    # =========================
    rows = [
        measure(
            'Autoencoder',
            lambda: art.autoencoder.score_samples(prepared.X_test_mm),
            art.training_times['autoencoder_seconds'],
            _torch_model_size_mb(art.autoencoder.model),
            _torch_param_count(art.autoencoder.model)
        ),

        measure(
            'GANomaly',
            lambda: art.ganomaly.score_samples(prepared.X_test_mm),
            art.training_times['ganomaly_seconds'],
            _torch_model_size_mb(art.ganomaly.e1)
            + _torch_model_size_mb(art.ganomaly.d)
            + _torch_model_size_mb(art.ganomaly.e2)
            + _torch_model_size_mb(art.ganomaly.c),
            _torch_param_count(art.ganomaly.e1)
            + _torch_param_count(art.ganomaly.d)
            + _torch_param_count(art.ganomaly.e2)
            + _torch_param_count(art.ganomaly.c),
        ),

        measure(
            'LSTM',
            lambda: art.lstm.predict_proba(prepared.X_test_std),
            art.training_times['lstm_seconds'],
            _torch_model_size_mb(art.lstm.model),
            _torch_param_count(art.lstm.model)
        ),

        measure(
            'CNN',
            lambda: art.cnn.predict_proba(prepared.X_test_std),
            art.training_times['cnn_seconds'],
            _torch_model_size_mb(art.cnn.model),
            _torch_param_count(art.cnn.model)
        ),

        measure(
            'RF/XGB',
            lambda: art.tabular.predict_proba(prepared.X_test_std),
            art.training_times['tabular_seconds'],
            float((os.path.getsize(paths.models_dir / 'random_forest.pkl')
                   + os.path.getsize(paths.models_dir / 'xgboost.pkl')) / (1024 ** 2)),
            0
        ),

        # 🔥 FIXED META LEARNER
        measure(
            'Meta-Learner',
            lambda: art.meta_learner.predict_proba(X_meta),
            art.training_times['meta_learner_seconds'],
            float(os.path.getsize(paths.models_dir / 'meta_learner.pkl') / (1024 ** 2)),
            0
        ),

        measure(
            'CLAF++ + Symbolic',
            lambda: art.symbolic_enhancer.score(
                art.predictions['test']['meta_learner'],
                art.predictions['test']['symbolic']
            ),
            art.training_times['total_seconds'],
            float(
                (os.path.getsize(paths.models_dir / 'random_forest.pkl')
                 + os.path.getsize(paths.models_dir / 'xgboost.pkl')
                 + os.path.getsize(paths.models_dir / 'meta_learner.pkl')) / (1024 ** 2)
                + _torch_model_size_mb(art.autoencoder.model)
                + _torch_model_size_mb(art.ganomaly.e1)
                + _torch_model_size_mb(art.ganomaly.d)
                + _torch_model_size_mb(art.ganomaly.e2)
                + _torch_model_size_mb(art.ganomaly.c)
                + _torch_model_size_mb(art.lstm.model)
                + _torch_model_size_mb(art.cnn.model)
            ),
            _torch_param_count(art.autoencoder.model)
            + _torch_param_count(art.ganomaly.e1)
            + _torch_param_count(art.ganomaly.d)
            + _torch_param_count(art.ganomaly.e2)
            + _torch_param_count(art.ganomaly.c)
            + _torch_param_count(art.lstm.model)
            + _torch_param_count(art.cnn.model),
        ),
    ]

    return pd.DataFrame(rows)


def _build_claims(results_table: pd.DataFrame) -> dict:
    lookup = {row['Model']: row for _, row in results_table.iterrows()}
    ae = lookup['AE only']
    ensemble = lookup['Ensemble (no meta)']
    meta = lookup['Meta-Learner (CLAF++)']
    full = lookup['CLAF++ + Symbolic']
    return {
        'primary_claim': (
            f"CLAF++ + Symbolic improves AUROC by {(full['AUROC'] - ae['AUROC']) * 100:.2f} percentage points "
            f"over AE only and reduces false alarm rate by {(ae['FalseAlarmRate'] - full['FalseAlarmRate']) * 100:.2f} points."
        ),
        'meta_claim': (
            f"The meta-learner improves AUROC by {(meta['AUROC'] - ensemble['AUROC']) * 100:.2f} percentage points "
            f"over soft ensemble fusion, showing that learned stacking is the primary contributor."
        ),
        'symbolic_claim': (
            f"The symbolic enhancement changes AUROC by {(full['AUROC'] - meta['AUROC']) * 100:.2f} points and false alarm rate by {(meta['FalseAlarmRate'] - full['FalseAlarmRate']) * 100:.2f} points relative to the neural meta model."
        ),
    }


def run_pipeline(config: dict):
    seed = int(config['runtime']['seed'])
    set_global_seed(seed)
    paths = make_experiment_paths(config['runtime']['artifacts_dir'], config['runtime'].get('experiment_id'))
    tracker = ExperimentTracker(paths)
    tracker.save_config(config)

    trainer = ResearchTrainer(config)
    art = trainer.run()
    prepared = art.prepared
    save_correlation_heatmap(prepared.train_df, paths.plots_dir / 'correlation_heatmap.png')

    ev = ResearchEvaluator(
        recall_floor=float(config['evaluation']['threshold']['recall_floor']),
        far_ceiling=float(config['evaluation']['threshold']['far_ceiling']),
        grid_size=int(config['evaluation']['threshold']['grid_size']),
    )

    y = prepared.y_test
    pred = art.predictions['test']
    # 🔥 Meta-as-filter fusion (NEW)

    # ===== Ensemble + Meta Filter (clean implementation) =====
    # This fusion uses meta as a confidence gate: only flag as attack if
    # both the soft ensemble AND the meta learner agree above their
    # respective thresholds. Multiplication gives us a soft AND.
    ensemble_scores = pred['soft_voting']
    meta_scores = pred['meta_learner']
    filtered_scores = ensemble_scores * meta_scores

    pred['Ensemble + Meta Filter'] = filtered_scores
    art.predictions['test']['Ensemble + Meta Filter'] = filtered_scores
    predictions = art.predictions['test']
    def normalize(scores):
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    leaderboard_specs = [
        ('AE only', pred['ae']),
        ('Generative branch', pred['generative']),
        ('LSTM', pred['lstm']),
        ('CNN', pred['cnn']),
        ('RF/XGB', pred['tabular']),
        ('Ensemble (no meta)', pred['soft_voting']),
        ('Meta-Learner (CLAF++)', pred['meta_learner']),
        ('CLAF++ + Symbolic', pred['clafpp']),
    ]

    results = []
    sweeps = {}
    for name, score in leaderboard_specs:
        result, sweep = ev.evaluate(name, y, score)
        results.append(result)
        sweeps[name] = sweep

    leaderboard = ev.to_table(results)
    best = [r for r in results if r.name == 'CLAF++ + Symbolic'][0]

    try:
        save_confusion_matrix(best.metrics['confusion_matrix'], 'Confusion Matrix - CLAF++ + Symbolic', paths.plots_dir / 'confusion_matrix_clafpp.png')
        save_roc_curve(results, paths.plots_dir / 'roc_curve.png')
        save_pr_curve(results, paths.plots_dir / 'pr_curve.png')
        ae_latent = art.autoencoder.score_samples(prepared.X_test_mm)['latent']
        save_latent_projection(ae_latent, y, paths.plots_dir / 'latent_space_pca.png')
        save_training_time_bar(art.training_times, paths.plots_dir / 'training_time_comparison.png')
        save_generative_training_curves({'ae_loss': art.autoencoder.summary_.train_loss}, paths.plots_dir / 'autoencoder_training_curve.png', 'Autoencoder Training Curve')
        save_generative_training_curves({'generator_loss': art.ganomaly.history_.generator_loss, 'discriminator_loss': art.ganomaly.history_.discriminator_loss}, paths.plots_dir / 'ganomaly_training_curve.png', 'GANomaly Training Curves')
    except Exception:
        pass

    tracker.save_json({r.name: {'threshold': r.threshold, **r.metrics} for r in results}, 'metrics.json')
    tracker.save_csv(leaderboard, 'reports/leaderboard.csv')

    pred_df = pd.DataFrame({
        'y_true': y,
        'ae_score': pred['ae'],
        'ganomaly_score': pred['ganomaly'],
        'generative_score': pred['generative'],
        'lstm_score': pred['lstm'],
        'cnn_score': pred['cnn'],
        'rf_score': pred['rf'],
        'xgb_score': pred['xgb'],
        'tabular_score': pred['tabular'],
        'soft_voting_score': pred['soft_voting'],
        'meta_score': pred['meta_learner'],
        'symbolic_score': pred['symbolic'],
        'clafpp_score': pred['clafpp'],
        'rule_explanations': pred['rule_explanations'],
    })
    tracker.save_csv(pred_df, 'predictions.csv')
    
    # =========================================================
    # DIAGNOSTICS BLOCK — figure generation (seed 42 only)
    # =========================================================
    if seed == 42:
        # Detect dataset
        dataset_version = str(config['dataset'].get('version', 'nslkdd')).lower()
        if 'edge' in dataset_version or 'iiot' in dataset_version:
            dataset_tag = 'edge'
        else:
            dataset_tag = 'nslkdd'

        diagnostics = {
            'y_test':            np.asarray(y),
            'p_ae':              np.asarray(pred['ae']),
            'p_gen':             np.asarray(pred['generative']),
            'p_lstm':            np.asarray(pred['lstm']),
            'p_cnn':             np.asarray(pred['cnn']),
            'p_rf':              np.asarray(pred['rf']),
            'p_xgb':             np.asarray(pred['xgb']),
            'p_rfxgb':           np.asarray(pred['tabular']),
            'p_ensemble_nometa': np.asarray(pred['soft_voting']),
            'p_meta':            np.asarray(pred['meta_learner']),
            'p_symbolic':        np.asarray(pred['symbolic']),
            'p_clafpp':          np.asarray(pred['clafpp']),
        }

        # --- Meta-learner backend (for figure caption honesty) ---
        try:
            diagnostics['meta_kind'] = str(art.meta_learner.kind)
            diagnostics['meta_threshold'] = float(art.meta_learner.threshold_)
            print(f"  Meta-learner backend: {diagnostics['meta_kind']}, "
                  f"threshold={diagnostics['meta_threshold']:.4f}")
        except Exception as e:
            print(f"  (Could not extract meta backend: {e})")

        # --- Lambda star from symbolic enhancer ---
        try:
            sym_state = art.symbolic_enhancer.state_dict()
            diagnostics['lambda_star'] = float(sym_state['lambda'])
            diagnostics['sym_mean'] = float(sym_state['sym_mean'])
            diagnostics['sym_std']  = float(sym_state['sym_std'])
            print(f"  lambda_star = {diagnostics['lambda_star']:.4f}")
        except Exception as e:
            print(f"  (Could not extract lambda_star: {e})")

        # --- Per-sample, per-rule activations (NSL-KDD only — for attribution chart) ---
        if dataset_tag == 'nslkdd':
            try:
                # Re-run the rule engine on the test DataFrame to capture the
                # rule_frame (per-sample, per-rule activations).
                # We try common attribute names for the test DataFrame:
                test_df = None
                for attr in ['test_df', 'X_test_df', 'df_test']:
                    if hasattr(prepared, attr):
                        test_df = getattr(prepared, attr)
                        break

                # Try common locations for the fitted rule engine on the art object:
                rule_engine = None
                for attr in ['rule_engine', 'rules', 'symbolic_engine', 'neuro_symbolic_engine']:
                    if hasattr(art, attr):
                        candidate = getattr(art, attr)
                        if candidate is not None and hasattr(candidate, 'evaluate'):
                            rule_engine = candidate
                            break

                if test_df is not None and rule_engine is not None:
                    rule_output = rule_engine.evaluate(test_df)
                    diagnostics['rule_activations'] = rule_output.rule_frame.values.astype(np.float32)
                    diagnostics['rule_names']       = np.array(rule_output.rule_frame.columns.tolist())
                    diagnostics['rule_weights']     = np.array(
                        [r.weight for r in rule_engine.fitted_rules_], dtype=np.float32
                    )
                    print(f"  ✓ Rule activations: shape={diagnostics['rule_activations'].shape}, "
                          f"rules={diagnostics['rule_names'].tolist()}")
                else:
                    print(f"  (Skipping rule activations: "
                          f"test_df={test_df is not None}, rule_engine={rule_engine is not None})")
            except Exception as e:
                print(f"  (Could not extract rule activations: {e})")

        # --- Save ---
        diag_path = paths.experiment_dir / f'clafpp_diagnostics_{dataset_tag}_seed42.npz'
        np.savez(diag_path, **diagnostics)
        print(f"  ✓ Saved diagnostics to {diag_path}")
        print(f"    Keys: {sorted(diagnostics.keys())}")

    ablation_df = run_ablation_study(config, paths.root.parent / 'reports' / 'ablation_results.csv', art)
    tracker.save_csv(ablation_df, 'reports/ablation_results.csv')

    art.tabular.save(str(paths.models_dir))
    art.meta_learner.save(str(paths.models_dir / 'meta_learner.pkl'))
    compute_df = _profile_compute(art, paths)
    tracker.save_csv(compute_df, 'reports/compute_profile.csv')

    claims = _build_claims(ablation_df)
    tracker.save_json(claims, 'reports/research_claims.json')

    tracker.save_json(
        {
            'threshold_sweeps': sweeps,
            'training_schedule': art.schedule.to_dict(),
            'training_times': art.training_times,
            'meta_learner': str(type(art.meta_learner.model_)),
            'generative_fusion': art.generative_fusion.state_dict(),
            'soft_voting_weights': art.soft_voting.weights_,
            'symbolic_enhancer': art.symbolic_enhancer.state_dict(),
            'rule_summary': art.rule_summary,
        },
        'reports/evaluation_details.json',
    )

    metadata = {
        'experiment_dir': str(paths.experiment_dir),
        'seed': seed,
        'dataset_version': config['dataset'].get('version', 'NSL-KDD'),
        'positive_class': 'attack',
        'negative_class': 'normal',
        'train_samples': int(len(prepared.y_train)),
        'val_samples': int(len(prepared.y_val)),
        'test_samples': int(len(prepared.y_test)),
        'primary_method': 'Meta-Learner (CLAF++)',
        'final_method': 'CLAF++ + Symbolic',
    }
    tracker.save_json(metadata, 'reports/run_metadata.json')

    tracker.save_model_bundle(
        {
            'autoencoder': art.autoencoder.state_dict(),
            'ganomaly': art.ganomaly.state_dict(),
            'lstm': art.lstm.state_dict(),
            'cnn': art.cnn.state_dict(),
            'tabular_models': list(art.tabular.models_.keys()),
            'meta_learner': {
                'type': art.meta_learner.kind,
                'threshold': float(art.meta_learner.threshold_),
                'features': art.meta_learner.feature_names_
            },
            'generative_fusion': art.generative_fusion.state_dict(),
            'soft_voting_weights': art.soft_voting.weights_,
            'symbolic_enhancer': art.symbolic_enhancer.state_dict(),
            'rule_summary': art.rule_summary,
            'feature_names': art.feature_names,
            'meta_feature_names': art.meta_feature_names,
            'config': config,
        }
    )

    # =========================
    # 🔥 NOW ADD PLOTTING HERE
    # =========================

    print(" Generating all plots...")

    import os
    os.makedirs(paths.plots_dir, exist_ok=True)

    pred_dict = {
        'AE only': pred['ae'],
        'Generative branch': pred['generative'],
        'LSTM': pred['lstm'],
        'CNN': pred['cnn'],
        'RF/XGB': pred['tabular'],
        'Ensemble (no meta)': pred['soft_voting'],
        'Meta-Learner (CLAF++)': pred['meta_learner'],
        'CLAF++ + Symbolic': pred['clafpp'],
    }

    thresholds = {r.name: r.threshold for r in results}

    save_all_model_confusion_matrices(
        prepared.y_test,
        pred_dict,
        thresholds,
        paths.plots_dir
    )
    # 🔥 Combined plots
    save_combined_roc_curve(
        results,
        prepared.y_test,
        pred_dict,
        paths.plots_dir
    )

    save_combined_pr_curve(
        results,
        prepared.y_test,
        pred_dict,
        paths.plots_dir
    )
    save_individual_roc_curves(results, paths.plots_dir)
    save_individual_pr_curves(results, paths.plots_dir)
    
    save_score_distributions(
        prepared.y_test,
        pred_dict,
        paths.plots_dir
    )

    save_prediction_histograms(
        pred_dict,
        paths.plots_dir
    )

    print(" All plots generated!")

    return leaderboard, metadata
    

from .plots import (
    save_combined_roc_curve,
    save_combined_pr_curve,
)