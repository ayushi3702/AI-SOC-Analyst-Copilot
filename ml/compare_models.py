import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from ml.data import load_cfg, load_frame
from ml.evaluate import (
    false_positive_rate_at_threshold,
    recall_at_threshold,
    threshold_for_fpr,
)
from ml.features import select_features
from ml.model_artifact import load_model_artifact, save_model_artifact
from ml.splits import get_split, train_validation_split


MLP_PARAMETERS = {
    "hidden_layer_sizes": [64, 32],
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 2048,
    "learning_rate_init": 0.001,
    "max_iter": 25,
    "early_stopping": False,
}

def _metrics(y_true, scores, threshold, families=None):
    y = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores)
    predicted = scores > threshold
    attacks = y == 1
    benign = ~attacks
    true_positives = int(np.sum(predicted & attacks))
    false_positives = int(np.sum(predicted & benign))
    family_recall = {}
    if families is not None:
        family_values = np.asarray(families)
        for family in sorted(set(family_values[attacks])):
            family_attack_mask = attacks & (family_values == family)
            family_recall[str(family)] = float(
                np.mean(predicted[family_attack_mask])
            )
    return {
        "threshold": float(threshold),
        "recall": recall_at_threshold(y, scores, threshold),
        "false_positive_rate": false_positive_rate_at_threshold(
            y, scores, threshold
        ),
        "precision": (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        ),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "alerts": int(predicted.sum()),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "family_recall": family_recall,
    }


def compare_models(
    df,
    cfg,
    xgb_artifact,
    split="unseen",
    target_fpr=0.01,
    with_dst_port=False,
    mlp_max_iter=25,
    xgb_artifact_sha256=None,
):
    """Compare the saved XGBoost baseline with an MLP on shared partitions."""
    initial_train, test = get_split(split, df, cfg)
    train, validation = train_validation_split(
        df, initial_train, seed=cfg["seed"]
    )
    if "event_id" in df.columns:
        train = df.loc[train, "event_id"].sort_values(kind="stable").index

    if xgb_artifact["split"] != split:
        raise ValueError(
            f"XGBoost artifact split {xgb_artifact['split']!r} does not match "
            f"requested split {split!r}"
        )
    if xgb_artifact["seed"] != cfg["seed"]:
        raise ValueError("XGBoost artifact seed does not match the requested split")
    if not np.isclose(xgb_artifact["target_fpr"], target_fpr):
        raise ValueError(
            "XGBoost artifact target FPR does not match the requested target FPR"
        )

    feature_names = list(xgb_artifact["feature_names"])
    if ("dst_port" in feature_names) != with_dst_port:
        raise ValueError(
            "XGBoost artifact destination-port feature does not match the "
            "requested feature configuration"
        )
    X_train = select_features(df.loc[train], feature_names=feature_names)
    X_validation = select_features(df.loc[validation], feature_names=feature_names)
    X_test = select_features(df.loc[test], feature_names=feature_names)
    y_train = df.loc[train, "is_attack"].astype("int8")
    y_validation = df.loc[validation, "is_attack"].astype("int8")
    y_test = df.loc[test, "is_attack"].astype("int8")
    sample_weights = compute_sample_weight("balanced", y_train)
    batch_size = min(MLP_PARAMETERS["batch_size"], len(X_train))

    xgb = xgb_artifact["model"]
    xgb_validation_scores = xgb.predict_proba(X_validation)[:, 1]
    xgb_threshold = xgb_artifact["threshold"]
    xgb_test_scores = xgb.predict_proba(X_test)[:, 1]

    mlp = Pipeline([
        ("scale", StandardScaler()),
        (
            "classifier",
            MLPClassifier(
                **{
                    **MLP_PARAMETERS,
                    "batch_size": batch_size,
                    "max_iter": mlp_max_iter,
                },
                random_state=cfg["seed"],
            ),
        ),
    ])
    with threadpool_limits(limits=1):
        mlp.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weights,
        )
    mlp_validation_scores = mlp.predict_proba(X_validation)[:, 1]
    mlp_threshold = threshold_for_fpr(
        y_validation, mlp_validation_scores, fpr=target_fpr
    )
    mlp_test_scores = mlp.predict_proba(X_test)[:, 1]
    mlp_classifier = mlp.named_steps["classifier"]

    test_families = df.loc[test, "family"] if "family" in df.columns else None
    xgb_params = xgb.get_params()
    return {
        "metadata": {
            "dataset_rows": int(len(df)),
            "split": split,
            "train_days": cfg.get("train_days", []),
            "test_days": cfg.get("test_days", []),
            "seed": int(cfg["seed"]),
            "validation_fraction": 0.2,
            "validation_split_grouping": "complete time blocks",
            "target_validation_fpr": float(target_fpr),
            "destination_port_included": bool(with_dst_port),
            "feature_count": len(feature_names),
            "xgboost_source": "saved_model_artifact",
            "xgboost_artifact_sha256": xgb_artifact_sha256,
            "feature_names_sha256": hashlib.sha256(
                json.dumps(feature_names, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "rows": {
                "fit": int(len(train)),
                "validation": int(len(validation)),
                "test": int(len(test)),
            },
        },
        "models": {
            "xgboost": {
                "parameters": {
                    name: xgb_params[name]
                    for name in (
                        "n_estimators",
                        "max_depth",
                        "learning_rate",
                        "tree_method",
                        "random_state",
                        "n_jobs",
                    )
                },
                "validation_false_positive_rate": false_positive_rate_at_threshold(
                    y_validation, xgb_validation_scores, xgb_threshold
                ),
                "metrics": _metrics(
                    y_test, xgb_test_scores, xgb_threshold, test_families
                ),
            },
            "mlp": {
                "parameters": {
                    **MLP_PARAMETERS,
                    "batch_size": batch_size,
                    "max_iter": mlp_max_iter,
                    "random_state": int(cfg["seed"]),
                    "class_weighting": "balanced per-example sample weights",
                    "scaling": "StandardScaler fit on model-training rows only",
                },
                "epochs_trained": int(mlp_classifier.n_iter_),
                "reached_iteration_limit": bool(
                    mlp_classifier.n_iter_ >= mlp_max_iter
                ),
                "validation_false_positive_rate": false_positive_rate_at_threshold(
                    y_validation, mlp_validation_scores, mlp_threshold
                ),
                "metrics": _metrics(
                    y_test, mlp_test_scores, mlp_threshold, test_families
                ),
            },
        },
        "artifacts": {
            "mlp_model": mlp,
            "feature_names": feature_names,
            "mlp_threshold": mlp_threshold,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare XGBoost with a small scaled neural network on shared "
            "time-based splits."
        )
    )
    parser.add_argument("--split", choices=["block", "unseen"], default="unseen")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--with-dst-port", action="store_true")
    parser.add_argument("--mlp-max-iter", type=int, default=25)
    parser.add_argument(
        "--xgb-model",
        default="ml/models/intrusion_unseen.joblib",
        help="Saved, validation-calibrated XGBoost baseline artifact.",
    )
    parser.add_argument(
        "--model-out",
        default="ml/models/intrusion_unseen_mlp.joblib",
        help="Path for the reusable MLP model artifact.",
    )
    parser.add_argument(
        "--report-out",
        default="evaluation/results/unseen_model_comparison.json",
        help="Path for the JSON comparison report.",
    )
    args = parser.parse_args()
    if not 0 <= args.target_fpr < 1:
        parser.error("--target-fpr must be in [0, 1)")
    if args.mlp_max_iter <= 0:
        parser.error("--mlp-max-iter must be positive")

    cfg = load_cfg()
    cfg["sample_frac"] = 1.0
    xgb_artifact_path = Path(args.xgb_model)
    xgb_artifact = load_model_artifact(xgb_artifact_path)
    xgb_artifact_sha256 = hashlib.sha256(
        xgb_artifact_path.read_bytes()
    ).hexdigest()
    if args.split == "unseen" and not args.with_dst_port:
        baseline_path = Path("evaluation/baselines/cicids2017_unseen_xgb.json")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        expected_hash = baseline["model"]["artifact_sha256_local_only"]
        if xgb_artifact_sha256 != expected_hash:
            parser.error(
                "The XGBoost artifact does not match the frozen unseen-day "
                f"baseline in {baseline_path}"
            )
    report = compare_models(
        load_frame(cfg),
        cfg,
        xgb_artifact,
        split=args.split,
        target_fpr=args.target_fpr,
        with_dst_port=args.with_dst_port,
        mlp_max_iter=args.mlp_max_iter,
        xgb_artifact_sha256=xgb_artifact_sha256,
    )
    artifact = report.pop("artifacts")
    save_model_artifact(
        model=artifact["mlp_model"],
        feature_names=artifact["feature_names"],
        threshold=artifact["mlp_threshold"],
        target_fpr=args.target_fpr,
        path=args.model_out,
        split=args.split,
        seed=cfg["seed"],
        validation_size=0.2,
    )

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"\nSaved MLP artifact to {args.model_out}")
    print(f"Saved comparison report to {report_path}")


if __name__ == "__main__":
    main()
