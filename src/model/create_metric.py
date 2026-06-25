import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score  # for AUC

def _as_multihot_target(target: torch.Tensor, C: int) -> torch.Tensor:
    """
    Accept target as [B] or [B,C], unify to [B,C] 0/1 float
    """
    if target.dim() == 1:
        return F.one_hot(target.long(), num_classes=C).float()
    elif target.dim() == 2 and target.size(1) == C:
        return target.float()
    else:
        raise ValueError(f"Unexpected target shape: {tuple(target.shape)}; expect [B] or [B,{C}]")

def _multilabel_bacc_factory(thresholds=None, eps=1e-6):
    """
    thresholds: None or list/tensor of length C (per-class thresholds)
    Returns a metric function: fn(logits, target) -> scalar
    """
    def _metric(logits, target):
        # Unify target shape
        B, C = logits.shape
        target = _as_multihot_target(target, C).to(logits.device)

        # Probabilities & thresholds
        probs = torch.sigmoid(logits)
        if thresholds is None or isinstance(thresholds, str):
            thr = 0.5
        else:
            thr = torch.tensor(thresholds, device=logits.device, dtype=probs.dtype).view(1, C)

        pred = (probs >= thr).to(target.dtype)

        # Per-class confusion
        TP = (pred * target).sum(dim=0)
        TN = ((1 - pred) * (1 - target)).sum(dim=0)
        FP = (pred * (1 - target)).sum(dim=0)
        FN = ((1 - pred) * target).sum(dim=0)

        TPR = TP / (TP + FN + eps)
        TNR = TN / (TN + FP + eps)
        bacc_per_class = 0.5 * (TPR + TNR)

        return bacc_per_class.mean()
    return _metric

def _multilabel_confusion(logits, target, thresholds=None, eps=1e-6):
    """
    Unified pipeline:
    - multi-hot target
    - Sigmoid + threshold
    - Per-class TP/TN/FP/FN
    """
    B, C = logits.shape
    target = _as_multihot_target(target, C).to(logits.device)

    probs = torch.sigmoid(logits)
    if thresholds is None or isinstance(thresholds, str):
        thr = 0.5
    else:
        thr = torch.tensor(thresholds, device=logits.device,
                           dtype=probs.dtype).view(1, C)

    pred = (probs >= thr).to(target.dtype)

    TP = (pred * target).sum(dim=0)
    TN = ((1 - pred) * (1 - target)).sum(dim=0)
    FP = (pred * (1 - target)).sum(dim=0)
    FN = ((1 - pred) * target).sum(dim=0)

    return TP, TN, FP, FN, probs, target

def _multilabel_f1_factory(thresholds=None, eps=1e-6):
    def _metric(logits, target):
        TP, TN, FP, FN, _, _ = _multilabel_confusion(
            logits, target, thresholds, eps
        )
        precision = TP / (TP + FP + eps)
        recall    = TP / (TP + FN + eps)
        f1_per_class = 2 * precision * recall / (precision + recall + eps)
        return f1_per_class.mean()
    return _metric

def _multilabel_kappa_factory(thresholds=None, eps=1e-6):
    def _metric(logits, target):
        TP, TN, FP, FN, _, _ = _multilabel_confusion(
            logits, target, thresholds, eps
        )
        n = TP + TN + FP + FN  # total samples per class

        po = (TP + TN) / (n + eps)

        # Expected agreement pe
        # Row/column sums: positive/negative marginal distributions
        p_yes_true  = (TP + FN) / (n + eps)
        p_yes_pred  = (TP + FP) / (n + eps)
        p_no_true   = (TN + FP) / (n + eps)
        p_no_pred   = (TN + FN) / (n + eps)

        pe = p_yes_true * p_yes_pred + p_no_true * p_no_pred

        kappa_per_class = (po - pe) / (1 - pe + eps)

        return kappa_per_class.mean()
    return _metric


def _multilabel_auc(logits, target, eps=1e-6):
    """
    Compute macro-average AUC using sklearn.metrics.roc_auc_score.
    """
    B, C = logits.shape
    target = _as_multihot_target(target, C).detach().cpu().numpy()
    probs  = torch.sigmoid(logits).detach().cpu().numpy()

    # If a class is all-0 or all-1 in labels, roc_auc_score will error;
    # Simple fix: on error, return average of valid classes (or 0.5).
    try:
        auc_macro = roc_auc_score(target, probs, average='macro')
    except ValueError:
        # Edge case: all positive or all negative, fallback to 0.5
        auc_macro = 0.5
    return float(auc_macro)

def _multilabel_final_factory(thresholds=None, eps=1e-6):
    bacc_fn  = _multilabel_bacc_factory(thresholds=thresholds, eps=eps)
    f1_fn    = _multilabel_f1_factory(thresholds=thresholds, eps=eps)
    kappa_fn = _multilabel_kappa_factory(thresholds=thresholds, eps=eps)

    def _metric(logits, target):
        # Compute each metric individually
        bacc  = bacc_fn(logits, target)          # torch scalar
        kappa = kappa_fn(logits, target)         # torch scalar
        f1    = f1_fn(logits, target)            # torch scalar
        auc   = _multilabel_auc(logits, target, eps=eps)  # python float

        # # Debug print (note: .item() only for torch.Tensor)
        # print(f"[metric] bACC={bacc.item():.4f}, "
        #       f"Kappa={kappa.item():.4f}, "
        #       f"F1={f1.item():.4f}, "
        #       f"AUC={auc:.4f}")

        final = (kappa + f1 + auc) / 3.0
        return final

    return _metric


def create_metric(name='acc', **kwargs):
    """
    metric function can be any function with scalar output.
    - For multi-label bACC/F1/Kappa/Final, may pass thresholds=[...]
    """
    if name == 'acc':
        return lambda logits, target: logits.argmax(dim=1).eq(target).float().mean()
    elif name == 'bacc':
        return _multilabel_bacc_factory(thresholds=kwargs.get('thresholds', None))
    elif name == 'f1':
        return _multilabel_f1_factory(thresholds=kwargs.get('thresholds', None))
    elif name == 'kappa':
        return _multilabel_kappa_factory(thresholds=kwargs.get('thresholds', None))
    elif name == 'auc':
        # NOTE: AUC is not recommended for frequent training calls; use only during evaluation
        return lambda logits, target: _multilabel_auc(logits, target)
    elif name == 'final':
        return _multilabel_final_factory(thresholds=kwargs.get('thresholds', None))
    else:
        raise NotImplementedError('Unknown metric name: %s' % name)
