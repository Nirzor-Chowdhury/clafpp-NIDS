from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve

def _safe_div(num: float, den: float) -> float: return float(num/den) if den else 0.0

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred=(y_prob>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    detection_rate=_safe_div(tp,tp+fn); false_alarm_rate=_safe_div(fp,fp+tn); specificity=_safe_div(tn,tn+fp)
    return {'threshold':float(threshold),'accuracy':float(accuracy_score(y_true,y_pred)),'precision':float(precision_score(y_true,y_pred,zero_division=0)),'recall':float(recall_score(y_true,y_pred,zero_division=0)),'f1':float(f1_score(y_true,y_pred,zero_division=0)),'auroc':float(roc_auc_score(y_true,y_prob)),'auprc':float(average_precision_score(y_true,y_prob)),'detection_rate':detection_rate,'false_alarm_rate':false_alarm_rate,'specificity':specificity,'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]]}

def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid_size: int = 101,
    recall_floor: float = 0.90,
    far_ceiling: float | None = None,
    model_name: str = None
):
    # 🔥 model-specific threshold ranges
    if model_name == 'Ensemble (no meta)':
        thresholds = np.linspace(0.08, 0.4, grid_size)   # 🔥 key fix
    elif model_name in ['LSTM', 'CNN']:
        thresholds = np.linspace(0.01, 0.3, grid_size)
    elif model_name == 'RF/XGB':
        thresholds = np.linspace(0.2, 0.9, grid_size)
    else:
        thresholds = np.linspace(0.05, 0.9, grid_size)

    results = []

    for t in thresholds:
        m = compute_binary_metrics(y_true, y_prob, float(t))

        m['meets_recall_floor'] = m['recall'] >= recall_floor
        m['meets_far_ceiling'] = (
            True if far_ceiling is None else m['false_alarm_rate'] <= far_ceiling
        )

        results.append(m)

    return results

def curve_payload(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    fpr,tpr,rt=roc_curve(y_true,y_prob); precision,recall,pt=precision_recall_curve(y_true,y_prob)
    return {'roc':{'fpr':fpr.tolist(),'tpr':tpr.tolist(),'thresholds':rt.tolist()}, 'pr':{'precision':precision.tolist(),'recall':recall.tolist(),'thresholds':pt.tolist()}}
def select_best_threshold(results):
    best_score = -1
    best_result = None

    for m in results:
        recall = m['recall']
        far = m['false_alarm_rate']

        # 🔥 IDS OBJECTIVE
        score = recall - 0.3 * far

        if score > best_score:
            best_score = score
            best_result = m

    return best_result