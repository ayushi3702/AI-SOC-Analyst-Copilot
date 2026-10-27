import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ml.compare_models import compare_models
from ml.evaluate import threshold_for_fpr
from ml.features import select_features
from ml.model_artifact import load_model_artifact, save_model_artifact
from ml.splits import get_split, train_validation_split


def _synthetic_data():
    rng = np.random.default_rng(42)
    rows_per_block = 40
    block_count = 8
    labels = np.tile([0, 1] * (rows_per_block // 2), block_count)
    frame = pd.DataFrame({
        "event_id": np.arange(len(labels)),
        "block": np.repeat(np.arange(block_count), rows_per_block),
        "day": np.repeat(["Monday"] * block_count, rows_per_block),
        "is_attack": labels,
        "signal": labels + rng.normal(0, 0.15, len(labels)),
        "noise": rng.normal(size=len(labels)),
    })
    frame["family"] = np.where(frame["is_attack"] == 1, "KnownAttack", "Benign")
    return frame


def _baseline_artifact(frame):
    cfg = {"seed": 42}
    initial_train, _ = get_split("block", frame, cfg)
    train, validation = train_validation_split(frame, initial_train, seed=42)
    X_train = select_features(frame.loc[train])
    feature_names = list(X_train.columns)
    X_validation = select_features(frame.loc[validation], feature_names=feature_names)
    y_train = frame.loc[train, "is_attack"]
    y_validation = frame.loc[validation, "is_attack"]
    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=1,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=compute_sample_weight("balanced", y_train),
    )
    return {
        "model": model,
        "feature_names": feature_names,
        "threshold": threshold_for_fpr(
            y_validation, model.predict_proba(X_validation)[:, 1], fpr=0.01
        ),
        "target_fpr": 0.01,
        "split": "block",
        "seed": 42,
    }


def test_model_comparison_uses_shared_partitions_and_validation_thresholds():
    frame = _synthetic_data()

    result = compare_models(
        frame,
        {"seed": 42},
        _baseline_artifact(frame),
        split="block",
        mlp_max_iter=3,
    )

    metadata = result["metadata"]
    assert metadata["rows"]["fit"] + metadata["rows"]["validation"] + metadata["rows"]["test"] == len(frame)
    assert metadata["feature_count"] == 2
    assert metadata["target_validation_fpr"] == 0.01
    assert metadata["xgboost_source"] == "saved_model_artifact"
    assert set(result["models"]) == {"xgboost", "mlp"}
    assert result["models"]["mlp"]["reached_iteration_limit"]
    for model in result["models"].values():
        metrics = model["metrics"]
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["false_positive_rate"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1
        assert "KnownAttack" in metrics["family_recall"]


def test_mlp_comparison_is_invariant_to_input_row_order():
    frame = _synthetic_data()
    artifact = _baseline_artifact(frame)
    original = compare_models(
        frame, {"seed": 42}, artifact, split="block", mlp_max_iter=3
    )
    shuffled = compare_models(
        frame.sample(frac=1, random_state=17),
        {"seed": 42},
        artifact,
        split="block",
        mlp_max_iter=3,
    )

    assert (
        original["models"]["mlp"]["metrics"]
        == shuffled["models"]["mlp"]["metrics"]
    )


def test_mlp_model_can_be_saved_with_same_inference_artifact_contract(tmp_path):
    frame = _synthetic_data()
    result = compare_models(
        frame,
        {"seed": 42},
        _baseline_artifact(frame),
        split="block",
        mlp_max_iter=3,
    )
    model_info = result["artifacts"]
    path = tmp_path / "mlp.joblib"

    save_model_artifact(
        model=model_info["mlp_model"],
        feature_names=model_info["feature_names"],
        threshold=model_info["mlp_threshold"],
        target_fpr=0.01,
        path=path,
        split="block",
        seed=42,
    )

    artifact = load_model_artifact(path)
    assert artifact["feature_names"] == model_info["feature_names"]
    assert artifact["threshold"] == model_info["mlp_threshold"]
    assert artifact["model"].predict_proba(
        pd.DataFrame({"signal": [0.0], "noise": [0.0]})[
            artifact["feature_names"]
        ]
    ).shape == (1, 2)
