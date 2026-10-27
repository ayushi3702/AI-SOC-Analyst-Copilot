import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "evaluation" / "baselines"


def _read_snapshot(name):
    with (BASELINES / name).open(encoding="utf-8") as snapshot_file:
        return json.load(snapshot_file)


def test_unseen_baseline_metrics_and_partition_counts_are_consistent():
    baseline = _read_snapshot("cicids2017_unseen_xgb.json")
    split = baseline["split"]
    metrics = baseline["test_metrics"]

    assert split["rows"]["fit"] + split["rows"]["validation"] + split["rows"]["test"] == 2827677
    assert metrics["true_positives"] + metrics["false_negatives"] == metrics["attacks"]
    assert metrics["true_positives"] + metrics["false_positives"] == metrics["alerts"]
    assert metrics["events"] == split["rows"]["test"]
    assert baseline["model"]["destination_port_included"] is False


def test_coverage_snapshot_totals_match_model_baseline():
    baseline = _read_snapshot("cicids2017_unseen_xgb.json")
    coverage = _read_snapshot("cicids2017_unseen_coverage.json")
    families = coverage["families"]
    metrics = baseline["test_metrics"]

    assert sum(
        family["fit_rows"] for family in families
    ) == baseline["split"]["rows"]["fit"]
    assert sum(
        family["validation_rows"] for family in families
    ) == baseline["split"]["rows"]["validation"]
    assert sum(
        family["test_rows"] for family in families
    ) == metrics["events"]
    assert sum(
        family["test_rows"] for family in families if family["family"] != "Benign"
    ) == metrics["attacks"]
    assert coverage["summary"]["missing_families"] == [
        "Bot", "DDoS", "PortScan", "Web"
    ]
