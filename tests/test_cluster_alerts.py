import json

import pandas as pd
import pytest

from ml.cluster_alerts import cluster_alerts, write_incidents


def _events():
    return pd.DataFrame({
        "event_id": ["e3", "e2", "e5", "e1", "e4", "e6"],
        "timestamp": [
            "2026-01-01T00:30:00Z",
            "2026-01-01T00:20:00Z",
            "2026-01-01T00:05:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:31:00Z",
            "2026-01-01T00:10:00Z",
        ],
        "src_ip": [
            "192.0.2.1", "192.0.2.1", "198.51.100.2",
            "192.0.2.1", "192.0.2.1", "192.0.2.1",
        ],
        "score": [0.9, 0.8, 0.95, 0.7, 0.99, 0.1],
        "is_alert": [True, True, True, True, True, False],
    })


def test_clustering_is_deterministic_and_preserves_alert_evidence():
    events = _events()

    incidents = cluster_alerts(events, window_minutes=30)
    reordered = cluster_alerts(events.sample(frac=1, random_state=12), window_minutes=30)

    assert incidents["incident_id"].tolist() == reordered["incident_id"].tolist()
    assert incidents["event_ids"].tolist() == reordered["event_ids"].tolist()
    assert incidents["event_ids"].tolist() == [["e1", "e2", "e3"], ["e5"], ["e4"]]
    assert incidents["alert_count"].tolist() == [3, 1, 1]
    assert incidents["max_score"].tolist() == [0.9, 0.95, 0.99]
    assert incidents["event_ids"].explode().is_unique


def test_fixed_window_does_not_chain_past_first_event():
    events = pd.DataFrame({
        "event_id": ["e1", "e2", "e3"],
        "timestamp": [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:20:00Z",
            "2026-01-01T00:40:00Z",
        ],
        "src_ip": ["192.0.2.1"] * 3,
        "score": [0.8, 0.9, 0.95],
        "is_alert": [True] * 3,
    })

    incidents = cluster_alerts(events, window_minutes=30)

    assert incidents["event_ids"].tolist() == [["e1", "e2"], ["e3"]]


def test_composite_entity_and_activity_gap_create_separate_incidents():
    events = pd.DataFrame({
        "event_id": ["e1", "e2", "e3", "e4"],
        "timestamp": [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:04:00Z",
            "2026-01-01T00:10:00Z",
            "2026-01-01T00:10:00Z",
        ],
        "src_ip": ["192.0.2.1", "192.0.2.1", "192.0.2.1", "192.0.2.2"],
        "dst_ip": ["198.51.100.1", "198.51.100.1", "198.51.100.1", "198.51.100.1"],
        "score": [0.8, 0.9, 0.95, 0.99],
        "is_alert": [True] * 4,
    })

    incidents = cluster_alerts(
        events,
        entity_columns=["src_ip", "dst_ip"],
        window_minutes=30,
        max_gap_minutes=5,
    )

    assert incidents["event_ids"].tolist() == [["e1", "e2"], ["e3"], ["e4"]]
    assert incidents["entity"].tolist() == [
        '["192.0.2.1","198.51.100.1"]',
        '["192.0.2.1","198.51.100.1"]',
        '["192.0.2.2","198.51.100.1"]',
    ]


def test_cluster_requires_valid_event_evidence_and_window():
    events = _events()
    with pytest.raises(ValueError, match="window_minutes"):
        cluster_alerts(events, window_minutes=0)
    with pytest.raises(ValueError, match="max_gap_minutes"):
        cluster_alerts(events, max_gap_minutes=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        cluster_alerts(events, max_gap_minutes=31, window_minutes=30)

    events.loc[0, "event_id"] = None
    with pytest.raises(ValueError, match="event_id"):
        cluster_alerts(events)

    events = _events()
    events.loc[0, "timestamp"] = "not-a-time"
    with pytest.raises(ValueError, match="timestamps"):
        cluster_alerts(events)


def test_no_alerts_returns_empty_incident_table():
    events = _events()
    events["is_alert"] = False

    incidents = cluster_alerts(events)

    assert incidents.empty
    assert "incident_id" in incidents


def test_incident_writer_preserves_event_ids_in_parquet_and_csv(tmp_path):
    incidents = cluster_alerts(_events(), window_minutes=30)
    parquet_path = tmp_path / "incidents.parquet"
    csv_path = tmp_path / "incidents.csv"

    write_incidents(incidents, parquet_path)
    write_incidents(incidents, csv_path)

    parquet_events = pd.read_parquet(parquet_path)["event_ids"].map(list).tolist()
    assert parquet_events == incidents["event_ids"].tolist()
    csv_events = pd.read_csv(csv_path)["event_ids"].map(json.loads).tolist()
    assert csv_events == incidents["event_ids"].tolist()
