import numpy as np
from sklearn.metrics import roc_auc_score

KEYS = ("auc", "acc", "f1")


def score(y, p):
    y = np.asarray(y).reshape(-1).astype(np.int64)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid probabilities or label count")
    if not len(y):
        return dict.fromkeys(KEYS)
    h = p > 0.5
    tp = int(((y == 1) & h).sum())
    fp = int(((y == 0) & h).sum())
    fn = int(((y == 1) & ~h).sum())
    both = len(np.unique(y)) == 2
    return dict(
        auc=float(roc_auc_score(y, p)) if both else None,
        acc=float((h == y).mean()),
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    )
