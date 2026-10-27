import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.robustness_report import build_robustness_report


def _dataset():
    rng = np.random.default_rng(42)
    rows_per_block = 40
    block_count = 8
    labels = np.tile([0, 1] * (rows_per_block // 2), block_count)
    signal = labels + rng.normal(0, 0.2, len(labels))
    frame = pd.DataFrame({
        "event_id": np.arange(len(labels)),
        "block": np.repeat(np.arange(block_count), rows_per_block),
        "day": "Monday",
        "is_attack": labels,
        "family": np.where(labels == 1, "KnownAttack", "Benign"),
        "signal": signal,
        "noise": rng.normal(size=len(labels)),
    })
    return frame


def _artifact(frame):
    features = frame[["signal", "noise"]]
    model = LogisticRegression(random_state=42).fit(
        features, frame["is_attack"]
    )
    return {
        "model": model,
        "feature_names": ["signal", "noise"],
        "threshold": 0.5,
        "split": "block",
    }


def test_robustness_report_holds_threshold_fixed_and_is_repeatable():
    frame = _dataset()
    artifact = _artifact(frame)
    arguments = {
        "df": frame,
        "cfg": {"seed": 42},
        "artifact": artifact,
        "split": "block",
        "noise_fractions": (0.01, 0.05),
        "batch_size": 32,
        "seed": 7,
    }

    report = build_robustness_report(**arguments)
    repeated = build_robustness_report(**arguments)

    assert [scenario["noise_fraction_of_fit_feature_std"] for scenario in report["scenarios"]] == [
        0.0, 0.01, 0.05
    ]
    assert all(
        scenario["metrics"]["threshold"] == artifact["threshold"]
        for scenario in report["scenarios"]
    )
    assert report["scenarios"] == repeated["scenarios"]
    assert report["clean_test_metrics"]["recall"] == report["scenarios"][0]["metrics"]["recall"]
    assert report["method"]["test_rows"] > 0
    assert "random" in report["purpose"]


@pytest.mark.parametrize(
    ("noise_fractions", "batch_size", "message"),
    [
        ((), 32, "noise_fractions"),
        ((0.0,), 32, "noise_fractions"),
        ((float("nan"),), 32, "noise_fractions"),
        ((0.01,), 0, "batch_size"),
    ],
)
def test_robustness_report_rejects_invalid_settings(
    noise_fractions, batch_size, message
):
    frame = _dataset()
    with pytest.raises(ValueError, match=message):
        build_robustness_report(
            frame,
            {"seed": 42},
            _artifact(frame),
            split="block",
            noise_fractions=noise_fractions,
            batch_size=batch_size,
        )
