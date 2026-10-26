import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


INCIDENT_COLUMNS = [
    "incident_id",
    "entity",
    "event_ids",
    "first_seen",
    "last_seen",
    "alert_count",
    "max_score",
]


def cluster_alerts(
    scored_events,
    entity_column="src_ip",
    entity_columns=None,
    window_minutes=30,
    max_gap_minutes=None,
    score_column="score",
    alert_column="is_alert",
):
    """Group flagged events by entity, with a fixed maximum span per incident."""
    if not math.isfinite(window_minutes) or window_minutes <= 0:
        raise ValueError("window_minutes must be a positive finite number")
    if entity_columns is None:
        entity_columns = [entity_column]
    elif isinstance(entity_columns, str):
        entity_columns = [entity_columns]
    else:
        entity_columns = list(entity_columns)
    if not entity_columns or len(set(entity_columns)) != len(entity_columns):
        raise ValueError("entity_columns must contain unique column names")
    if max_gap_minutes is not None:
        if not math.isfinite(max_gap_minutes) or max_gap_minutes <= 0:
            raise ValueError("max_gap_minutes must be a positive finite number")
        if max_gap_minutes > window_minutes:
            raise ValueError("max_gap_minutes cannot exceed window_minutes")
    required = {
        "event_id", "timestamp", score_column, alert_column, *entity_columns
    }
    missing = required - set(scored_events.columns)
    if missing:
        raise ValueError(f"Scored events are missing fields: {sorted(missing)}")
    if (
        scored_events["event_id"].isna().any()
        or scored_events["event_id"].duplicated().any()
    ):
        raise ValueError("event_id values must be present and unique")
    for column in entity_columns:
        if scored_events[column].isna().any():
            raise ValueError(f"{column} values must be present")
        if scored_events[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} values must not be empty")
    if scored_events[alert_column].isna().any():
        raise ValueError(f"{alert_column} values must be present")
    if not scored_events[alert_column].isin([True, False, 0, 1]).all():
        raise ValueError(f"{alert_column} values must be boolean")

    events = scored_events.copy()
    events["parsed_timestamp"] = pd.to_datetime(
        events["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    events["numeric_score"] = pd.to_numeric(events[score_column], errors="coerce")
    if events["parsed_timestamp"].isna().any():
        raise ValueError("Event timestamps must be valid")
    if not np.isfinite(events["numeric_score"]).all():
        raise ValueError("Event scores must be finite numbers")
    events = events.loc[events[alert_column].astype(bool)].copy()
    if len(entity_columns) == 1:
        events["entity_key"] = events[entity_columns[0]].astype(str)
    else:
        events["entity_key"] = events[entity_columns].astype(str).apply(
            lambda row: json.dumps(row.tolist(), separators=(",", ":")), axis=1
        )
    events["event_key"] = events["event_id"].astype(str)
    events = events.sort_values(
        ["entity_key", "parsed_timestamp", "event_key"], kind="mergesort"
    )

    window = pd.Timedelta(minutes=window_minutes)
    max_gap = (
        pd.Timedelta(minutes=max_gap_minutes)
        if max_gap_minutes is not None
        else None
    )
    incidents = []
    for entity, group in events.groupby("entity_key", sort=True):
        current_rows = []
        current_start = None
        previous_timestamp = None
        for row in group.itertuples(index=False):
            timestamp = row.parsed_timestamp
            exceeded_window = current_rows and timestamp > current_start + window
            exceeded_gap = (
                current_rows
                and max_gap is not None
                and timestamp > previous_timestamp + max_gap
            )
            if exceeded_window or exceeded_gap:
                incidents.append(_make_incident(entity, current_rows))
                current_rows = []
            if not current_rows:
                current_start = timestamp
            current_rows.append((row.event_key, timestamp, row.numeric_score))
            previous_timestamp = timestamp
        if current_rows:
            incidents.append(_make_incident(entity, current_rows))

    if not incidents:
        return pd.DataFrame(columns=INCIDENT_COLUMNS)
    result = pd.DataFrame(incidents, columns=INCIDENT_COLUMNS)
    return result.sort_values(
        ["first_seen", "entity", "incident_id"], kind="mergesort"
    ).reset_index(drop=True)


def _make_incident(entity, rows):
    first_seen = rows[0][1]
    last_seen = rows[-1][1]
    event_ids = [row[0] for row in rows]
    identity = f"{entity}\0{first_seen.isoformat()}\0{event_ids[0]}"
    incident_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return {
        "incident_id": incident_id,
        "entity": entity,
        "event_ids": event_ids,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "alert_count": len(rows),
        "max_score": max(row[2] for row in rows),
    }


def write_incidents(incidents, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        incidents.to_parquet(output, index=False)
    elif output.suffix.lower() == ".csv":
        csv_incidents = incidents.copy()
        csv_incidents["event_ids"] = csv_incidents["event_ids"].map(json.dumps)
        csv_incidents.to_csv(output, index=False)
    else:
        raise ValueError("Output path must end in .parquet or .csv")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster scored alert events into deterministic incidents."
    )
    parser.add_argument("--input", required=True, help="Scored CSV or Parquet file.")
    parser.add_argument(
        "--output",
        default="data/processed/incidents.parquet",
        help="Output path ending in .parquet or .csv.",
    )
    parser.add_argument(
        "--entity-columns", "--entity-column", nargs="+", default=["src_ip"]
    )
    parser.add_argument("--window-minutes", type=float, default=30)
    parser.add_argument("--max-gap-minutes", type=float)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".parquet":
        events = pd.read_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        events = pd.read_csv(input_path)
    else:
        parser.error("--input path must end in .parquet or .csv")
    incidents = cluster_alerts(
        events,
        entity_columns=args.entity_columns,
        window_minutes=args.window_minutes,
        max_gap_minutes=args.max_gap_minutes,
    )
    write_incidents(incidents, args.output)
    print(f"Created {len(incidents):,} incidents from {len(events):,} scored events.")
    print(f"Saved incidents to {args.output}")


if __name__ == "__main__":
    main()
