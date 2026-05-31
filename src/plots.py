from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

sns.set_theme(style='whitegrid')


# =========================
# GLOBAL PLOTS
# =========================
def save_roc_curve(results, output_path):
    plt.figure(figsize=(6,5))
    for r in results:
        plt.plot(
            r.curves['roc']['fpr'],
            r.curves['roc']['tpr'],
            label=f"{r.name} ({r.metrics.get('auroc', 0):.3f})"
        )
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (All Models)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path,dpi=180)
    plt.close()


def save_pr_curve(results, output_path):
    plt.figure(figsize=(6,5))
    for r in results:
        plt.plot(
            r.curves['pr']['recall'],
            r.curves['pr']['precision'],
            label=f"{r.name} ({r.metrics.get('auprc', 0):.3f})"
        )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve (All Models)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_path,dpi=180)
    plt.close()


# =========================
# 🔥 PER MODEL PLOTS
# =========================

def _safe_name(name):
    return name.replace("/", "_").replace(" ", "_")


def save_all_model_confusion_matrices(y_true, predictions_dict, threshold_dict, output_dir):
    for model_name, scores in predictions_dict.items():
        threshold = threshold_dict.get(model_name, 0.5)
        y_pred = (scores >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal','Attack'],
                    yticklabels=['Normal','Attack'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()

        plt.savefig(f"{output_dir}/confusion_{_safe_name(model_name)}.png", dpi=180)
        plt.close()


def save_individual_roc_curves(results, output_dir):
    for r in results:
        plt.figure(figsize=(5,4))
        plt.plot(r.curves['roc']['fpr'], r.curves['roc']['tpr'])
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC - {r.name}')
        plt.tight_layout()

        plt.savefig(f"{output_dir}/roc_{_safe_name(r.name)}.png", dpi=180)
        plt.close()


def save_individual_pr_curves(results, output_dir):
    for r in results:
        plt.figure(figsize=(5,4))
        plt.plot(r.curves['pr']['recall'], r.curves['pr']['precision'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve - {r.name}')
        plt.tight_layout()

        plt.savefig(f"{output_dir}/pr_{_safe_name(r.name)}.png", dpi=180)
        plt.close()


def save_score_distributions(y_true, predictions_dict, output_dir):
    for model_name, scores in predictions_dict.items():
        df = pd.DataFrame({
            'score': scores,
            'label': np.where(y_true == 1, 'Attack', 'Normal')
        })

        plt.figure(figsize=(6,4))
        sns.kdeplot(data=df, x='score', hue='label', fill=True)
        plt.title(f'Score Distribution - {model_name}')
        plt.tight_layout()

        plt.savefig(f"{output_dir}/score_dist_{_safe_name(model_name)}.png", dpi=180)
        plt.close()


def save_prediction_histograms(predictions_dict, output_dir):
    for model_name, scores in predictions_dict.items():
        plt.figure(figsize=(6,4))
        plt.hist(scores, bins=50)
        plt.title(f'Prediction Histogram - {model_name}')
        plt.tight_layout()

        plt.savefig(f"{output_dir}/hist_{_safe_name(model_name)}.png", dpi=180)
        plt.close()


# =========================
# 🔥 COMBINED CURVES (FIXED)
# =========================

def save_combined_roc_curve(results, y_true, predictions, output_dir):
    plt.figure()

    for r in results:
        model_name = r.name
        scores = predictions.get(model_name, None)
        if scores is None:
            continue

        fpr, tpr, _ = roc_curve(y_true, scores)

        auc = r.metrics.get('auroc', 0)

        if "CLAF++" in model_name:
            plt.plot(fpr, tpr, linewidth=3,
                     label=f"{model_name} (AUC={auc:.3f})")
        else:
            plt.plot(fpr, tpr, alpha=0.6,
                     label=f"{model_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (All Models)")
    plt.legend()
    plt.grid()

    plt.savefig(f"{output_dir}/roc_all_models.png", dpi=200)
    plt.close()


def save_combined_pr_curve(results, y_true, predictions, output_dir):
    plt.figure()

    for r in results:
        model_name = r.name
        scores = predictions.get(model_name, None)
        if scores is None:
            continue

        precision, recall, _ = precision_recall_curve(y_true, scores)

        auprc = r.metrics.get('auprc', 0)

        if "CLAF++" in model_name:
            plt.plot(recall, precision, linewidth=3,
                     label=f"{model_name} (AUPRC={auprc:.3f})")
        else:
            plt.plot(recall, precision, alpha=0.6,
                     label=f"{model_name} (AUPRC={auprc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (All Models)")
    plt.legend()
    plt.grid()

    plt.savefig(f"{output_dir}/pr_all_models.png", dpi=200)
    plt.close()


# =========================
# LATENT SPACE
# =========================
def save_latent_projection(latent, labels, output_path):
    proj = PCA(n_components=2).fit_transform(latent)
    df = pd.DataFrame({
        'x': proj[:,0],
        'y': proj[:,1],
        'label': np.where(labels==1,'Attack','Normal')
    })

    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df,x='x',y='y',hue='label')
    plt.tight_layout()
    plt.savefig(output_path,dpi=180)
    plt.close()


# =========================
# TRAINING TIME
# =========================
def save_training_time_bar(training_times, output_path):
    rows = [(k.replace('_seconds',''),v) for k,v in training_times.items() if k.endswith('_seconds')]
    df = pd.DataFrame(rows,columns=['Stage','Seconds']).sort_values('Seconds',ascending=False)

    plt.figure(figsize=(8,4))
    plt.barh(df['Stage'], df['Seconds'])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path,dpi=180)
    plt.close()


def save_correlation_heatmap(df, output_path):
    corr = df.select_dtypes(include=[float, int]).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()