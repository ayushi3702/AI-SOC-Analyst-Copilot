import argparse
from pathlib import Path

import duckdb
import numpy as np

from ml.data import load_cfg
from ml.features import select_features
from ml.model_artifact import load_model_artifact


def load_event_features(db_path):
    """Read event context and model features without joining the labels table."""
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return connection.execute("""
            SELECT e.event_id, e.timestamp, e.src_ip, e.dst_ip, e.dst_port,
                   f.* EXCLUDE (event_id, source_port, destination_port)
            FROM events e
            JOIN flow_features f USING (event_id)
        """).df()


def score_events(events, artifact):
    required_metadata = {"event_id", "timestamp", "src_ip", "dst_ip"}
    missing = required_metadata - set(events.columns)
    if missing:
        raise ValueError(f"Input events are missing fields: {sorted(missing)}")
    if events["event_id"].isna().any() or events["event_id"].duplicated().any():
        raise ValueError("event_id values must be present and unique")

    features = select_features(
        events, feature_names=artifact["feature_names"]
    )
    scores = artifact["model"].predict_proba(features)[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite scores")

    scored = events[["event_id", "timestamp", "src_ip", "dst_ip"]].copy()
    scored["score"] = scores
    scored["is_alert"] = scored["score"] > artifact["threshold"]
    return scored


def write_scored_events(events, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        events.to_parquet(output, index=False)
    elif output.suffix.lower() == ".csv":
        events.to_csv(output, index=False)
    else:
        raise ValueError("Output path must end in .parquet or .csv")


def main():
    parser = argparse.ArgumentParser(
        description="Score DuckDB events with a saved binary intrusion model."
    )
    parser.add_argument(
        "--model",
        default="ml/models/intrusion_binary.joblib",
        help="Path to a trusted model artifact.",
    )
    parser.add_argument(
        "--db",
        help="DuckDB path; defaults to db_path in configs/settings.yaml.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/scored_events.parquet",
        help="Output path ending in .parquet or .csv.",
    )
    args = parser.parse_args()

    cfg = load_cfg()
    db_path = args.db or cfg["db_path"]
    artifact = load_model_artifact(args.model)
    scored = score_events(load_event_features(db_path), artifact)
    write_scored_events(scored, args.output)
    print(f"Scored {len(scored):,} events; {int(scored['is_alert'].sum()):,} alerts.")
    print(f"Saved scores to {args.output}")


if __name__ == "__main__":
    main()
