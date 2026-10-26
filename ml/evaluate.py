import numpy as np, pandas as pd

def threshold_for_recall(y_val, s_val, recall=0.95):
    """Pick the score threshold on VALIDATION data, so the test number is honest."""
    y, s = np.asarray(y_val), np.asarray(s_val)
    return np.quantile(s[y == 1], 1 - recall)


def threshold_for_fpr(y_val, s_val, fpr=0.01):
    """Calibrate an anomaly threshold from validation benign scores only."""
    y, s = np.asarray(y_val), np.asarray(s_val)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same shape")
    if not 0 <= fpr < 1:
        raise ValueError("fpr must be in [0, 1)")
    benign_scores = s[y == 0]
    if benign_scores.size == 0:
        raise ValueError("validation data must include benign examples")
    return float(np.quantile(benign_scores, 1 - fpr))


def recall_at_threshold(y_true, scores, threshold):
    """Measure attack recall on a scored set without recalibrating its threshold."""
    y, s = np.asarray(y_true), np.asarray(scores)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same shape")
    attacks = y == 1
    if not attacks.any():
        raise ValueError("evaluation data must include attack examples")
    return float(np.mean(s[attacks] > threshold))


def false_positive_rate_at_threshold(y_true, scores, threshold):
    """Measure benign alert rate at a fixed, previously calibrated threshold."""
    y, s = np.asarray(y_true), np.asarray(scores)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same shape")
    benign = y == 0
    if not benign.any():
        raise ValueError("evaluation data must include benign examples")
    return float(np.mean(s[benign] > threshold))


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
