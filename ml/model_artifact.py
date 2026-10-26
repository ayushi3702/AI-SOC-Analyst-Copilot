import math
from pathlib import Path

import joblib


ARTIFACT_VERSION = 1


def save_model_artifact(
    model,
    feature_names,
    threshold,
    target_fpr,
    path,
    split,
    seed,
    validation_size=0.2,
):
    if not feature_names:
        raise ValueError("model artifact must include at least one feature")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be in (0, 1)")

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "task": "binary_intrusion",
        "model": model,
        "feature_names": list(feature_names),
        "threshold": float(threshold),
        "threshold_comparison": ">",
        "target_fpr": float(target_fpr),
        "validation_size": float(validation_size),
        "split": split,
        "seed": int(seed),
    }
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    return artifact


def load_model_artifact(path):
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must contain a dictionary")
    required = {
        "artifact_version",
        "task",
        "model",
        "feature_names",
        "threshold",
        "threshold_comparison",
        "target_fpr",
        "validation_size",
        "split",
        "seed",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing fields: {sorted(missing)}")
    if artifact["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported model artifact version: {artifact['artifact_version']}"
        )
    if artifact["task"] != "binary_intrusion":
        raise ValueError(f"Unsupported model artifact task: {artifact['task']}")
    if artifact["threshold_comparison"] != ">":
        raise ValueError("Unsupported model threshold comparison")
    if not artifact["feature_names"]:
        raise ValueError("Model artifact has no features")
    if not hasattr(artifact["model"], "predict_proba"):
        raise ValueError("Model artifact model does not support predict_proba")
    return artifact
