import pandas as pd
import pytest
import duckdb
from sklearn.linear_model import LogisticRegression

from ml.infer import load_event_features, score_events
from ml.model_artifact import load_model_artifact, save_model_artifact


def _artifact():
    model = LogisticRegression(random_state=0)
    model.fit(
        pd.DataFrame(
            {
                "feature_a": [0.0, 1.0, 0.1, 0.9],
                "feature_b": [1.0, 0.0, 0.9, 0.1],
            }
        ),
        [0, 1, 0, 1],
    )
    return {
        "model": model,
        "feature_names": ["feature_a", "feature_b"],
        "threshold": 0.5,
    }


def test_artifact_round_trip_persists_model_features_and_calibration(tmp_path):
    path = tmp_path / "models" / "detector.joblib"
    model = _artifact()["model"]

    save_model_artifact(
        model,
        ["feature_a", "feature_b"],
        threshold=0.42,
        target_fpr=0.01,
        path=path,
        split="block",
        seed=7,
    )

    loaded = load_model_artifact(path)
    assert loaded["feature_names"] == ["feature_a", "feature_b"]
    assert loaded["threshold"] == pytest.approx(0.42)
    assert loaded["target_fpr"] == pytest.approx(0.01)
    assert loaded["validation_size"] == pytest.approx(0.2)
    assert loaded["split"] == "block"


def test_score_events_aligns_saved_features_and_uses_threshold():
    events = pd.DataFrame({
        "event_id": ["event-1", "event-2"],
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "src_ip": ["192.0.2.1", "192.0.2.2"],
        "dst_ip": ["198.51.100.1", "198.51.100.2"],
        "feature_b": [1.0, 0.0],
        "feature_a": [0.0, 1.0],
    })

    scored = score_events(events, _artifact())

    assert scored["event_id"].tolist() == ["event-1", "event-2"]
    assert scored["score"].iloc[0] < 0.5
    assert scored["score"].iloc[1] > 0.5
    assert scored["is_alert"].tolist() == [False, True]


def test_score_events_rejects_missing_features_and_duplicate_event_ids():
    events = pd.DataFrame({
        "event_id": ["event-1"],
        "timestamp": ["2026-01-01"],
        "src_ip": ["192.0.2.1"],
        "dst_ip": ["198.51.100.1"],
        "feature_a": [0.1],
    })
    with pytest.raises(ValueError, match="model features"):
        score_events(events, _artifact())

    events["feature_b"] = [0.9]
    events.loc[1] = events.loc[0]
    with pytest.raises(ValueError, match="unique"):
        score_events(events, _artifact())


def test_artifact_rejects_invalid_calibration_values(tmp_path):
    with pytest.raises(ValueError, match="target_fpr"):
        save_model_artifact(
            _artifact()["model"],
            ["feature_a"],
            threshold=0.5,
            target_fpr=1.0,
            path=tmp_path / "invalid.joblib",
            split="block",
            seed=1,
        )


def test_duckdb_event_loader_does_not_read_ground_truth_labels(tmp_path):
    database = tmp_path / "events.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("""
            CREATE TABLE events (
                event_id VARCHAR, timestamp TIMESTAMP, src_ip VARCHAR,
                dst_ip VARCHAR, dst_port INTEGER
            )
        """)
        connection.execute("""
            CREATE TABLE flow_features (
                event_id VARCHAR, source_port INTEGER, destination_port INTEGER,
                feature_a DOUBLE, feature_b DOUBLE
            )
        """)
        connection.execute("""
            CREATE TABLE labels (event_id VARCHAR, label VARCHAR)
        """)
        connection.execute("""
            INSERT INTO events VALUES
                ('e1', '2026-01-01 00:00:00', '192.0.2.1', '198.51.100.1', 443)
        """)
        connection.execute("""
            INSERT INTO flow_features VALUES ('e1', 1234, 443, 0.1, 0.9)
        """)
        connection.execute("INSERT INTO labels VALUES ('e1', 'secret-ground-truth')")

    events = load_event_features(database)

    assert events["event_id"].tolist() == ["e1"]
    assert "label" not in events.columns
    assert "feature_a" in events.columns
