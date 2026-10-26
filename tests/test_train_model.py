import numpy as np
import pandas as pd

from ml.model_artifact import load_model_artifact
from ml.train_model import train_binary_model


def test_training_saves_feature_list_and_validation_calibrated_threshold(tmp_path):
    rng = np.random.default_rng(21)
    labels = np.tile([0, 1] * 10, 12)
    frame = pd.DataFrame({
        "block": np.repeat(np.arange(12), 20),
        "is_attack": labels,
        "signal": labels + rng.normal(0, 0.1, len(labels)),
        "noise": rng.normal(size=len(labels)),
    })
    artifact_path = tmp_path / "detector.joblib"

    metrics = train_binary_model(
        frame,
        {"seed": 42},
        split="block",
        output_path=artifact_path,
    )

    artifact = load_model_artifact(artifact_path)
    assert artifact["feature_names"] == ["signal", "noise"]
    assert artifact["threshold"] == metrics["threshold"]
    assert artifact["target_fpr"] == 0.01
    assert metrics["n_train"] + metrics["n_validation"] + metrics["n_test"] == len(frame)
    assert 0 <= metrics["test_fpr"] <= 1


def test_destination_port_ablation_is_recorded_in_feature_list(tmp_path):
    rng = np.random.default_rng(7)
    labels = np.tile([0, 1] * 10, 8)
    frame = pd.DataFrame({
        "block": np.repeat(np.arange(8), 20),
        "is_attack": labels,
        "signal": labels + rng.normal(0, 0.2, len(labels)),
        "dst_port": np.where(labels, 4444, 443),
    })

    metrics = train_binary_model(
        frame,
        {"seed": 42},
        split="block",
        output_path=tmp_path / "with_port.joblib",
        with_dst_port=True,
    )

    from ml.model_artifact import load_model_artifact

    artifact = load_model_artifact(tmp_path / "with_port.joblib")
    assert metrics["with_dst_port"] is True
    assert "dst_port" in artifact["feature_names"]
