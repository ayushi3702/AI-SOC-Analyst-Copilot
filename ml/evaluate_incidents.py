import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from ml.cluster_alerts import cluster_alerts
from ml.data import DAY_MAP, load_cfg


def load_test_ground_truth(db_path, test_days):
    """Load only offline labels for the configured held-out days."""
    day_prefixes = [day[:3].lower() for day in test_days]
    placeholders = ", ".join("?" for _ in day_prefixes)
    query = f"""
        SELECT event_id, label, is_attack, day
        FROM labels
        WHERE lower(substr(trim(day), 1, 3)) IN ({placeholders})
    """
    with duckdb.connect(str(db_path), read_only=True) as connection:
        truth = connection.execute(query, day_prefixes).df()
    if truth.empty:
        raise ValueError(f"No ground-truth labels found for test days: {test_days}")
    return truth


def evaluate_scored_events(
    scored_events,
    test_ground_truth,
    families,
    test_days=("Thursday", "Friday"),
    window_minutes=30,
    entity_columns=("src_ip",),
    max_gap_minutes=None,
    sample_size=20,
):
    """Offline-only evaluation; labels are joined after inference and filtered before clustering."""
    if isinstance(entity_columns, str):
        entity_columns = [entity_columns]
    else:
        entity_columns = list(entity_columns)
    if not entity_columns:
        raise ValueError("entity_columns must include at least one column")
    scored_required = {
        "event_id", "timestamp", "score", "is_alert", *entity_columns
    }
    truth_required = {"event_id", "label", "is_attack", "day"}
    missing_scored = scored_required - set(scored_events.columns)
    missing_truth = truth_required - set(test_ground_truth.columns)
    if missing_scored:
        raise ValueError(f"Scored data is missing fields: {sorted(missing_scored)}")
    if missing_truth:
        raise ValueError(f"Ground truth is missing fields: {sorted(missing_truth)}")
    if scored_events["event_id"].isna().any() or scored_events["event_id"].duplicated().any():
        raise ValueError("Scored event_id values must be present and unique")
    if (
        test_ground_truth["event_id"].isna().any()
        or test_ground_truth["event_id"].duplicated().any()
    ):
        raise ValueError("Ground-truth event_id values must be present and unique")
    if not 0 < sample_size:
        raise ValueError("sample_size must be positive")

    truth = test_ground_truth.copy()
    truth["day"] = truth["day"].astype(str).map(
        lambda day: DAY_MAP.get(day.strip()[:3].lower(), day)
    )
    normalized_test_days = {
        DAY_MAP.get(day.strip()[:3].lower(), day) for day in test_days
    }
    truth = truth.loc[truth["day"].isin(normalized_test_days)].copy()
    if truth.empty:
        raise ValueError(f"No ground-truth labels found for test days: {test_days}")

    scores = scored_events.loc[
        scored_events["event_id"].isin(truth["event_id"])
    ].copy()
    evaluation = scores.merge(
        truth, on="event_id", how="inner", validate="one_to_one"
    )
    if len(evaluation) != len(truth):
        missing_ids = len(truth) - len(evaluation)
        raise ValueError(f"Scored data is missing {missing_ids} test event(s)")
    if evaluation.empty:
        raise ValueError("No scored test events were found")

    evaluation["is_attack"] = pd.to_numeric(
        evaluation["is_attack"], errors="coerce"
    )
    if evaluation["is_attack"].isna().any() or not evaluation["is_attack"].isin([0, 1]).all():
        raise ValueError("Ground-truth is_attack values must be binary")
    evaluation["is_attack"] = evaluation["is_attack"].astype(bool)
    if evaluation["is_alert"].isna().any() or not evaluation["is_alert"].isin(
        [True, False, 0, 1]
    ).all():
        raise ValueError("Scored is_alert values must be boolean")
    evaluation["is_alert"] = evaluation["is_alert"].astype(bool)
    attacks = evaluation["is_attack"]
    alerts = evaluation["is_alert"]
    true_positives = int((attacks & alerts).sum())
    false_positives = int((~attacks & alerts).sum())
    false_negatives = int((attacks & ~alerts).sum())
    benign_count = int((~attacks).sum())
    alert_count = int(alerts.sum())
    event_metrics = {
        "test_days": sorted(normalized_test_days),
        "test_events": int(len(evaluation)),
        "test_attacks": int(attacks.sum()),
        "test_benign": benign_count,
        "alerts": alert_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "recall": true_positives / int(attacks.sum()) if attacks.any() else None,
        "false_positive_rate": false_positives / benign_count if benign_count else None,
        "precision": true_positives / alert_count if alert_count else None,
        "alert_rate": alert_count / len(evaluation),
        "cluster_entity": entity_columns,
        "cluster_window_minutes": float(window_minutes),
        "cluster_max_gap_minutes": max_gap_minutes,
    }

    evaluation["family"] = evaluation["label"].map(families).fillna("Other")
    attack_rows = evaluation.loc[evaluation["is_attack"]]
    family_metrics = (
        attack_rows.groupby("family", sort=True)
        .agg(
            attack_events=("event_id", "size"),
            detected=("is_alert", "sum"),
        )
        .reset_index()
    )
    family_metrics["missed"] = family_metrics["attack_events"] - family_metrics["detected"]
    family_metrics["recall"] = family_metrics["detected"] / family_metrics["attack_events"]

    test_alerts = evaluation.loc[evaluation["is_alert"]].copy()
    incidents = cluster_alerts(
        test_alerts,
        entity_columns=entity_columns,
        window_minutes=window_minutes,
        max_gap_minutes=max_gap_minutes,
    )
    truth_by_id = evaluation.set_index("event_id")
    if incidents.empty:
        incidents["attack_count"] = pd.Series(dtype="int64")
        incidents["benign_count"] = pd.Series(dtype="int64")
        incidents["contains_attack"] = pd.Series(dtype="bool")
        incidents["contains_benign"] = pd.Series(dtype="bool")
        incidents["families"] = pd.Series(dtype="object")
    else:
        attack_counts = []
        benign_counts = []
        incident_families = []
        for event_ids in incidents["event_ids"]:
            rows = truth_by_id.loc[event_ids]
            attack_counts.append(int(rows["is_attack"].sum()))
            benign_counts.append(int((~rows["is_attack"]).sum()))
            incident_families.append(
                sorted(rows.loc[rows["is_attack"], "family"].unique().tolist())
            )
        incidents["attack_count"] = attack_counts
        incidents["benign_count"] = benign_counts
        incidents["contains_attack"] = incidents["attack_count"] > 0
        incidents["contains_benign"] = incidents["benign_count"] > 0
        incidents["families"] = incident_families

    sizes = incidents["alert_count"]
    incident_metrics = {
        "cluster_entity": entity_columns,
        "cluster_window_minutes": float(window_minutes),
        "cluster_max_gap_minutes": max_gap_minutes,
        "incident_count": int(len(incidents)),
        "clustered_alert_events": int(sizes.sum()) if len(sizes) else 0,
        "mean_alerts_per_incident": float(sizes.mean()) if len(sizes) else 0.0,
        "median_alerts_per_incident": float(sizes.median()) if len(sizes) else 0.0,
        "p90_alerts_per_incident": float(sizes.quantile(0.9)) if len(sizes) else 0.0,
        "max_alerts_per_incident": int(sizes.max()) if len(sizes) else 0,
        "incidents_with_attacks": int(incidents["contains_attack"].sum()),
        "benign_only_incidents": int(
            (incidents["attack_count"] == 0).sum()
        ),
        "mixed_attack_benign_incidents": int(
            (incidents["contains_attack"] & incidents["contains_benign"]).sum()
        ),
        "attack_alert_events_in_incidents": true_positives,
        "attack_event_incident_coverage": (
            true_positives / int(attacks.sum()) if attacks.any() else None
        ),
        "attack_alert_incident_coverage": (
            true_positives / int((attacks & alerts).sum())
            if (attacks & alerts).any()
            else None
        ),
    }
    samples = incidents.sort_values(
        ["contains_attack", "alert_count", "max_score"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(sample_size)
    return evaluation, event_metrics, family_metrics, incidents, incident_metrics, samples


def _write_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_incident_csv(incidents, path):
    output = incidents.copy()
    for column in ("event_ids", "families"):
        if column in output.columns:
            output[column] = output[column].map(json.dumps)
    output.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate scored events and incident clusters on held-out days."
    )
    parser.add_argument(
        "--scores",
        default="data/processed/scored_events_unseen.parquet",
        help="Full scored event CSV or Parquet file.",
    )
    parser.add_argument(
        "--db",
        help="DuckDB path; defaults to db_path from configs/settings.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/evaluation_unseen",
        help="Directory for JSON, CSV, and Parquet evaluation outputs.",
    )
    parser.add_argument("--window-minutes", type=float, default=30)
    parser.add_argument(
        "--entity-columns", nargs="+", default=["src_ip"]
    )
    parser.add_argument("--max-gap-minutes", type=float)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")

    cfg = load_cfg()
    scores_path = Path(args.scores)
    if scores_path.suffix.lower() == ".parquet":
        scored_events = pd.read_parquet(scores_path)
    elif scores_path.suffix.lower() == ".csv":
        scored_events = pd.read_csv(scores_path)
    else:
        parser.error("--scores path must end in .parquet or .csv")
    ground_truth = load_test_ground_truth(args.db or cfg["db_path"], cfg["test_days"])
    outputs = evaluate_scored_events(
        scored_events,
        ground_truth,
        cfg["families"],
        test_days=cfg["test_days"],
        window_minutes=args.window_minutes,
        entity_columns=args.entity_columns,
        max_gap_minutes=args.max_gap_minutes,
        sample_size=args.sample_size,
    )
    _, event_metrics, family_metrics, incidents, incident_metrics, samples = outputs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(event_metrics, output_dir / "event_metrics.json")
    _write_json(incident_metrics, output_dir / "incident_metrics.json")
    family_metrics.to_csv(output_dir / "family_metrics.csv", index=False)
    incidents.to_parquet(output_dir / "test_incidents.parquet", index=False)
    _write_incident_csv(incidents, output_dir / "test_incidents.csv")
    _write_incident_csv(samples, output_dir / "incident_samples.csv")

    print("Held-out event metrics:")
    print(json.dumps(event_metrics, indent=2))
    print("\nPer-family attack recall:")
    print(family_metrics.to_string(index=False))
    print("\nHeld-out incident metrics:")
    print(json.dumps(incident_metrics, indent=2))
    print(f"\nSaved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
