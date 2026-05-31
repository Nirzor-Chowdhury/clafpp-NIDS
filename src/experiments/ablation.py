from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..evaluation import ResearchEvaluator


def run_ablation_study(config, output_csv_path, run_artifacts):
    ev = ResearchEvaluator(
        recall_floor=float(config['evaluation']['threshold']['recall_floor']),
        far_ceiling=float(config['evaluation']['threshold']['far_ceiling']),
        grid_size=int(config['evaluation']['threshold']['grid_size']),
    )
    y = run_artifacts.prepared.y_test
    pred = run_artifacts.predictions['test']

    experiments = [
        ('AE only', pred['ae']),
        ('AE + LSTM', 0.5 * pred['ae'] + 0.5 * pred['lstm']),
        ('AE + CNN', 0.5 * pred['ae'] + 0.5 * pred['cnn']),
        ('Ensemble (no meta)', pred['soft_voting']),
        ('Meta-Learner (CLAF++)', pred['meta_learner']),
        ('CLAF++ + Symbolic', pred['clafpp']),
    ]

    rows = []
    for name, score in experiments:
        result, _ = ev.evaluate(name, y, score)
        rows.append(
            {
                'Model': name,
                'Accuracy': result.metrics['accuracy'],
                'Precision': result.metrics['precision'],
                'Recall': result.metrics['recall'],
                'F1': result.metrics['f1'],
                'AUROC': result.metrics['auroc'],
                'AUPRC': result.metrics['auprc'],
                'FalseAlarmRate': result.metrics['false_alarm_rate'],
                'DetectionRate': result.metrics['detection_rate'],
                'Threshold': result.threshold,
            }
        )

    df = pd.DataFrame(rows)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return df
