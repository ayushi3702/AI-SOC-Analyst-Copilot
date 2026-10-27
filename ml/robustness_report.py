import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.data import load_cfg, load_frame
from ml.evaluate import false_positive_rate_at_threshold, recall_at_threshold
from ml.features import select_features
from ml.model_artifact import load_model_artifact
from ml.splits import get_split, train_validation_split


NOISE_FRACTIONS = (0.01, 0.05, 0.10)
DEFAULT_BATCH_SIZE = 50_000


def _feature_scales(df, indices, feature_names, batch_size):
    sums = np.zeros(len(feature_names), dtype=np.float64)
    squared_sums = np.zeros(len(feature_names), dtype=np.float64)
    count = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        values = select_features(
            df.loc[batch_indices], feature_names=feature_names
        ).to_numpy(dtype=np.float64, copy=False)
        sums += values.sum(axis=0)
        squared_sums += np.square(values).sum(axis=0)
        count += len(values)
    if not count:
        raise ValueError("Training partition is empty")
    means = sums / count
    variances = np.maximum(squared_sums / count - np.square(means), 0.0)
    return np.sqrt(variances)


def _metrics(labels, scores, threshold, families):
    predicted = scores > threshold
    attacks = labels == 1
    benign = ~attacks
    true_positives = int(np.sum(predicted & attacks))
    false_positives = int(np.sum(predicted & benign))
    family_recall = {}
    for family in sorted(set(families[attacks])):
        mask = attacks & (families == family)
        family_recall[str(family)] = float(np.mean(predicted[mask]))
    return {
        "threshold": float(threshold),
        "recall": recall_at_threshold(labels, scores, threshold),
        "false_positive_rate": false_positive_rate_at_threshold(
            labels, scores, threshold
        ),
        "precision": (
            true_positives / int(predicted.sum()) if predicted.any() else 0.0
        ),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "alerts": int(predicted.sum()),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "family_recall": family_recall,
    }


def build_robustness_report(
    df,
    cfg,
    artifact,
    split="unseen",
    noise_fractions=NOISE_FRACTIONS,
    batch_size=DEFAULT_BATCH_SIZE,
    seed=42,
    artifact_sha256=None,
):
    """Measure fixed-threshold performance under non-adaptive feature noise."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not noise_fractions or any(
        not np.isfinite(level) or level <= 0 for level in noise_fractions
    ):
        raise ValueError("noise_fractions must contain positive finite values")
    if artifact["split"] != split:
        raise ValueError(
            f"Model artifact split {artifact['split']!r} does not match "
            f"requested split {split!r}"
        )

    initial_train, test = get_split(split, df, cfg)
    train, _ = train_validation_split(df, initial_train, seed=cfg["seed"])
    feature_names = list(artifact["feature_names"])
    train_indices = df.loc[train, "event_id"].sort_values(
        kind="stable"
    ).index.to_numpy()
    test_indices = df.loc[test, "event_id"].sort_values(
        kind="stable"
    ).index.to_numpy()
    test_frame = df.loc[test_indices, ["is_attack"]].copy()
    labels = test_frame["is_attack"].to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Test labels must be binary")
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError("Test partition must contain attack and benign rows")
    if "family" in df.columns:
        families = (
            df.loc[test_indices, "family"]
            .fillna("Unmapped")
            .astype(str)
            .to_numpy()
        )
    else:
        families = np.where(labels == 1, "Attack", "Benign")

    scales = _feature_scales(df, train_indices, feature_names, batch_size)
    threshold = float(artifact["threshold"])
    scenarios = [0.0, *sorted(set(float(level) for level in noise_fractions))]
    scores_by_scenario = {
        level: np.empty(len(test_indices), dtype=np.float64)
        for level in scenarios
    }
    rngs = {
        level: np.random.default_rng(seed)
        for level in scenarios
        if level > 0
    }
    for start in range(0, len(test_indices), batch_size):
        batch_indices = test_indices[start : start + batch_size]
        features_frame = select_features(
            df.loc[batch_indices], feature_names=feature_names
        )
        features = features_frame.to_numpy(dtype=np.float64, copy=False)
        end = start + len(batch_indices)
        scores_by_scenario[0.0][start:end] = artifact["model"].predict_proba(
            features_frame
        )[:, 1]
        for level in scenarios:
            if level == 0:
                continue
            noise = rngs[level].standard_normal(features.shape)
            perturbed = features_frame + noise * scales * level
            scores_by_scenario[level][start:end] = artifact[
                "model"
            ].predict_proba(perturbed)[:, 1]

    baseline = _metrics(labels, scores_by_scenario[0.0], threshold, families)
    results = []
    for level in scenarios:
        metrics = _metrics(labels, scores_by_scenario[level], threshold, families)
        results.append({
            "noise_fraction_of_fit_feature_std": level,
            "metrics": metrics,
            "recall_change_from_clean": metrics["recall"] - baseline["recall"],
            "pr_auc_change_from_clean": metrics["pr_auc"] - baseline["pr_auc"],
        })

    return {
        "report_name": "Non-adaptive feature-noise robustness stress test",
        "purpose": (
            "Offline sensitivity measurement using random, zero-mean numeric "
            "feature noise. This does not generate network traffic or model "
            "valid protocol transformations."
        ),
        "method": {
            "split": split,
            "seed": int(seed),
            "perturbation": (
                "Independent zero-mean Gaussian noise scaled by each feature's "
                "standard deviation on model-fit rows"
            ),
            "noise_fractions": list(scenarios[1:]),
            "threshold": threshold,
            "threshold_policy": "Saved artifact threshold held fixed for all scenarios",
            "fit_rows_for_scale": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "feature_count": len(feature_names),
            "batch_size": int(batch_size),
            "xgboost_artifact_sha256": artifact_sha256,
            "limitations": [
                "Synthetic feature corruption is not a valid-traffic simulation.",
                "Perturbations are random and non-adaptive; no attack strategy is evaluated.",
                "Results describe this dataset and fixed threshold, not production robustness.",
            ],
        },
        "clean_test_metrics": baseline,
        "scenarios": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure the saved detector's offline sensitivity to "
            "non-adaptive numeric feature noise."
        )
    )
    parser.add_argument(
        "--model",
        default="ml/models/intrusion_unseen.joblib",
        help="Path to the saved unseen-day XGBoost artifact.",
    )
    parser.add_argument("--split", choices=["block", "unseen"], default="unseen")
    parser.add_argument(
        "--output",
        default="evaluation/results/unseen_robustness.json",
        help="JSON report output path.",
    )
    args = parser.parse_args()

    cfg = load_cfg()
    cfg["sample_frac"] = 1.0
    artifact_path = Path(args.model)
    artifact = load_model_artifact(artifact_path)
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if args.split == "unseen":
        baseline_path = Path("evaluation/baselines/cicids2017_unseen_xgb.json")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        expected_hash = baseline["model"]["artifact_sha256_local_only"]
        if artifact_hash != expected_hash:
            parser.error(
                "The model artifact does not match the frozen unseen-day "
                f"baseline in {baseline_path}"
            )

    report = build_robustness_report(
        load_frame(cfg),
        cfg,
        artifact,
        split=args.split,
        seed=cfg["seed"],
        artifact_sha256=artifact_hash,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"\nSaved robustness report to {output}")


if __name__ == "__main__":
    main()
