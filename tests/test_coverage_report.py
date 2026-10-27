import json

import pandas as pd
import pytest

from ml.coverage_report import build_coverage_report


def _frame():
    return pd.DataFrame(
        {
            "day": [
                "Monday", "Monday", "Tuesday", "Tuesday",
                "Thursday", "Thursday", "Friday", "Friday",
            ],
            "block": [
                "mon-1", "mon-2", "tue-1", "tue-2",
                "thu-1", "thu-2", "fri-1", "fri-2",
            ],
            "label": [
                "BENIGN", "DDoS", "BENIGN", "DDoS",
                "BENIGN", "Bot", "BENIGN", "PortScan",
            ],
            "is_attack": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def _cfg():
    return {
        "seed": 42,
        "train_days": ["Monday", "Tuesday"],
        "test_days": ["Thursday", "Friday"],
        "families": {
            "BENIGN": "Benign",
            "DDoS": "DDoS",
            "Bot": "Bot",
            "PortScan": "PortScan",
        },
    }


def test_unseen_report_identifies_test_families_missing_from_training():
    report = build_coverage_report(_frame(), _cfg(), split="unseen")
    by_family = report.set_index("family")

    assert by_family.loc["Bot", "train_rows"] == 0
    assert by_family.loc["Bot", "test_attack_rows"] == 1
    assert by_family.loc["Bot", "missing_from_training"]
    assert by_family.loc["PortScan", "train_rows"] == 0
    assert by_family.loc["PortScan", "missing_from_training"]
    assert by_family.loc["DDoS", "train_attack_rows"] > 0
    assert not by_family.loc["DDoS", "missing_from_training"]
    assert json.loads(by_family.loc["Bot", "test_labels"]) == ["Bot"]


def test_coverage_counts_are_partitioned_across_all_rows():
    frame = _frame()
    report = build_coverage_report(frame, _cfg(), split="unseen")

    assert report["train_rows"].sum() + report["validation_rows"].sum() == 4
    assert report["test_rows"].sum() == 4
    assert (
        report["train_attack_rows"].sum()
        + report["validation_attack_rows"].sum()
        + report["test_attack_rows"].sum()
        == frame["is_attack"].sum()
    )


def test_unmapped_test_labels_are_reported_explicitly():
    frame = _frame()
    frame.loc[7, "label"] = "NewAttack"

    report = build_coverage_report(frame, _cfg(), split="unseen")
    unmapped = report.set_index("family").loc["Unmapped"]

    assert unmapped["unmapped_labels_in_test"]
    assert json.loads(unmapped["test_labels"]) == ["NewAttack"]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(), "missing columns"),
        (_frame().set_axis([0, 0, 1, 2, 3, 4, 5, 6]), "index values"),
        (_frame().assign(is_attack=2), "binary"),
    ],
)
def test_invalid_coverage_data_is_rejected(frame, message):
    with pytest.raises(ValueError, match=message):
        build_coverage_report(frame, _cfg(), split="unseen")
