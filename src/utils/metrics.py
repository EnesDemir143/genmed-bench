"""
Metrics modülü.

sklearn.metrics ile evaluation metrikleri hesaplar.

Kullanım:
    from src.utils.metrics import compute_all_metrics
    
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
from sklearn import metrics as skmetrics


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    multi_label: bool = False,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Tüm metrikleri hesaplar.
    
    Args:
        y_true: Ground truth labels [N] veya [N, C] (multi-label)
        y_pred: Predicted labels [N] veya [N, C]
        y_prob: Prediction probabilities [N, C] (AUC için)
        multi_label: Multi-label classification
        class_names: Class isimleri (report için)
        
    Returns:
        Dict with all metrics
    """
    results = {}
    
    if multi_label:
        results.update(_compute_multilabel_metrics(y_true, y_pred, y_prob))
    else:
        results.update(_compute_binary_multiclass_metrics(y_true, y_pred, y_prob))
    
    return results


def _compute_binary_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Binary veya multiclass classification metrikleri."""
    results = {}
    
    # Basic metrics
    results['accuracy'] = skmetrics.accuracy_score(y_true, y_pred)
    results['balanced_accuracy'] = skmetrics.balanced_accuracy_score(y_true, y_pred)
    
    # Class-wise metrics
    num_classes = len(np.unique(y_true))
    average = 'binary' if num_classes == 2 else 'macro'
    
    results['precision'] = skmetrics.precision_score(
        y_true, y_pred, average=average, zero_division=0
    )
    results['recall'] = skmetrics.recall_score(
        y_true, y_pred, average=average, zero_division=0
    )
    results['f1'] = skmetrics.f1_score(
        y_true, y_pred, average=average, zero_division=0
    )
    
    # Weighted versions
    results['precision_weighted'] = skmetrics.precision_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    results['recall_weighted'] = skmetrics.recall_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    results['f1_weighted'] = skmetrics.f1_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    results['confusion_matrix'] = skmetrics.confusion_matrix(y_true, y_pred).tolist()
    
    # AUC (requires probabilities)
    if y_prob is not None:
        # Tek sınıf kontrolü - AUC hesaplanamaz
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            # Sadece bir sınıf varsa AUC tanımsız, 0.5 ata
            results['roc_auc'] = 0.5
            results['average_precision'] = None
        else:
            try:
                if num_classes == 2:
                    # Binary: use positive class probability
                    prob_positive = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                    results['roc_auc'] = skmetrics.roc_auc_score(y_true, prob_positive)
                    results['average_precision'] = skmetrics.average_precision_score(
                        y_true, prob_positive
                    )
                else:
                    # Multiclass: OvR
                    results['roc_auc'] = skmetrics.roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='macro'
                    )
                    results['roc_auc_weighted'] = skmetrics.roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='weighted'
                    )
            except ValueError:
                # AUC hesaplanamaz (örn: tek sınıf varsa)
                results['roc_auc'] = 0.5
    
    # Per-class metrics
    results['per_class'] = {}
    for cls in np.unique(y_true):
        cls_mask = y_true == cls
        results['per_class'][int(cls)] = {
            'precision': skmetrics.precision_score(
                y_true == cls, y_pred == cls, zero_division=0
            ),
            'recall': skmetrics.recall_score(
                y_true == cls, y_pred == cls, zero_division=0
            ),
            'f1': skmetrics.f1_score(
                y_true == cls, y_pred == cls, zero_division=0
            ),
            'support': int(cls_mask.sum()),
        }
    
    return results


def _compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Multi-label classification metrikleri (NIH için)."""
    results = {}
    
    # Sample-wise accuracy
    results['accuracy'] = skmetrics.accuracy_score(y_true, y_pred)
    
    # Micro/macro averages
    results['precision_micro'] = skmetrics.precision_score(
        y_true, y_pred, average='micro', zero_division=0
    )
    results['recall_micro'] = skmetrics.recall_score(
        y_true, y_pred, average='micro', zero_division=0
    )
    results['f1_micro'] = skmetrics.f1_score(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    results['precision_macro'] = skmetrics.precision_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    results['recall_macro'] = skmetrics.recall_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    results['f1_macro'] = skmetrics.f1_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Hamming loss
    results['hamming_loss'] = skmetrics.hamming_loss(y_true, y_pred)
    
    # AUC per label
    if y_prob is not None:
        try:
            results['roc_auc_micro'] = skmetrics.roc_auc_score(
                y_true, y_prob, average='micro'
            )
            results['roc_auc_macro'] = skmetrics.roc_auc_score(
                y_true, y_prob, average='macro'
            )
            results['roc_auc_weighted'] = skmetrics.roc_auc_score(
                y_true, y_prob, average='weighted'
            )
            
            # Per-label AUC
            results['per_label_auc'] = {}
            for i in range(y_true.shape[1]):
                # Tek sınıf kontrolü
                unique_vals = np.unique(y_true[:, i])
                if len(unique_vals) < 2:
                    results['per_label_auc'][i] = 0.5
                else:
                    try:
                        auc = skmetrics.roc_auc_score(y_true[:, i], y_prob[:, i])
                        results['per_label_auc'][i] = auc
                    except ValueError:
                        results['per_label_auc'][i] = 0.5
                    
        except ValueError:
            results['roc_auc_micro'] = 0.5
            results['roc_auc_macro'] = 0.5
    
    return results


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """Sklearn classification report döner."""
    return skmetrics.classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )


# =============================================================================
# THRESHOLD OPTIMIZATION
# =============================================================================

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1'
) -> float:
    """
    Optimal threshold bulur.
    
    Args:
        y_true: Ground truth
        y_prob: Probabilities
        metric: Optimize edilecek metrik ('f1', 'accuracy', 'balanced_accuracy')
        
    Returns:
        Optimal threshold
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        if metric == 'f1':
            score = skmetrics.f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = skmetrics.accuracy_score(y_true, y_pred)
        elif metric == 'balanced_accuracy':
            score = skmetrics.balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh
