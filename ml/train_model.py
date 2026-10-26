import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ml.data import load_cfg, load_frame
from ml.evaluate import (
    false_positive_rate_at_threshold,
    recall_at_threshold,
    threshold_for_fpr,
)
from ml.features import select_features
from ml.model_artifact import save_model_artifact
from ml.splits import get_split, train_validation_split


def train_binary_model(
    df,
    cfg,
    split="block",
    output_path="ml/models/intrusion_binary.joblib",
    target_fpr=0.01,
    with_dst_port=False,
):
    initial_train, test = get_split(split, df, cfg)
    train, validation = train_validation_split(
        df, initial_train, seed=cfg["seed"]
    )

    X_train = select_features(df.loc[train], use_dst_port=with_dst_port)
    feature_names = list(X_train.columns)
    X_validation = select_features(
        df.loc[validation], feature_names=feature_names
    )
    X_test = select_features(df.loc[test], feature_names=feature_names)
    y_train = df.loc[train, "is_attack"]
    y_validation = df.loc[validation, "is_attack"]
    y_test = df.loc[test, "is_attack"]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=-1,
        random_state=cfg["seed"],
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=compute_sample_weight("balanced", y_train),
    )

    validation_scores = model.predict_proba(X_validation)[:, 1]
    threshold = threshold_for_fpr(
        y_validation, validation_scores, fpr=target_fpr
    )
    test_scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "threshold": threshold,
        "target_fpr": target_fpr,
        "test_recall": recall_at_threshold(y_test, test_scores, threshold),
        "test_fpr": false_positive_rate_at_threshold(
            y_test, test_scores, threshold
        ),
        "test_pr_auc": float(average_precision_score(y_test, test_scores)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "with_dst_port": with_dst_port,
    }
    if "family" in df.columns:
        test_families = df.loc[test, "family"]
        family_recall = {}
        for family in sorted(test_families.dropna().unique()):
            family_mask = test_families.eq(family).to_numpy()
            family_attacks = y_test.to_numpy()[family_mask] == 1
            if family_attacks.any():
                family_recall[str(family)] = float(
                    np.mean(test_scores[family_mask][family_attacks] > threshold)
                )
        metrics["family_recall"] = family_recall
    metric_names = (
        "threshold",
        "test_recall",
        "test_fpr",
        "test_pr_auc",
        "test_roc_auc",
    )
    if not np.isfinite([metrics[name] for name in metric_names]).all():
        raise ValueError("Model evaluation produced a non-finite metric")

    save_model_artifact(
        model=model,
        feature_names=feature_names,
        threshold=threshold,
        target_fpr=target_fpr,
        path=output_path,
        split=split,
        seed=cfg["seed"],
        validation_size=0.2,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train a binary detector and save a validation-calibrated artifact."
    )
    parser.add_argument("--split", choices=["block", "unseen"], required=True)
    parser.add_argument(
        "--model-out",
        default="ml/models/intrusion_binary.joblib",
        help="Path for the reusable model artifact.",
    )
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument(
        "--with-dst-port",
        action="store_true",
        help="Include destination port as a feature for a controlled ablation.",
    )
    args = parser.parse_args()
    if not 0 <= args.target_fpr < 1:
        parser.error("--target-fpr must be in [0, 1)")

    cfg = load_cfg()
    metrics = train_binary_model(
        load_frame(cfg),
        cfg,
        split=args.split,
        output_path=args.model_out,
        target_fpr=args.target_fpr,
        with_dst_port=args.with_dst_port,
    )
    print(metrics)
    print(f"Saved model artifact to {args.model_out}")


if __name__ == "__main__":
    main()
