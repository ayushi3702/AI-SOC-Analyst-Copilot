import numpy as np, pandas as pd

def threshold_for_recall(y_val, s_val, recall=0.95):
    """Pick the score threshold on VALIDATION data, so the test number is honest."""
    y, s = np.asarray(y_val), np.asarray(s_val)
    return np.quantile(s[y == 1], 1 - recall)

def alert_reduction(y_true, score, thr):
    y, s = np.asarray(y_true), np.asarray(score)
    flagged = s >= thr
    return {
        "recall": float(flagged[y == 1].mean()) if (y == 1).any() else np.nan,
        "precision": float(y[flagged].mean()) if flagged.any() else np.nan,
        "alerts_to_review": int(flagged.sum()),
        "total_events": len(y),
        "reduction": float(1 - flagged.mean()),   # share of events analysts never see
    }

def report(y_val, s_val, y_test, s_test, recalls=(0.90, 0.95, 0.99)):
    rows = []
    for r in recalls:
        thr = threshold_for_recall(y_val, s_val, r)
        rows.append({"target_recall": r, **alert_reduction(y_test, s_test, thr)})
    return pd.DataFrame(rows)
