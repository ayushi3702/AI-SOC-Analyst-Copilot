import duckdb
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.evaluate_incidents import evaluate_scored_events, load_test_ground_truth
from ml.infer import load_event_features, score_events, write_scored_events
from ml.model_artifact import load_model_artifact, save_model_artifact


def test_saved_model_runs_through_duckdb_inference_and_heldout_evaluation(tmp_path):
    database = tmp_path / "soc.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("""
            CREATE TABLE events (
                event_id VARCHAR,
                timestamp TIMESTAMP,
                src_ip VARCHAR,
                dst_ip VARCHAR,
                dst_port INTEGER
            )
        """)
        connection.execute("""
            CREATE TABLE flow_features (
                event_id VARCHAR,
                source_port INTEGER,
                destination_port INTEGER,
                feature_a DOUBLE,
                feature_b DOUBLE
            )
        """)
        connection.execute("""
            CREATE TABLE labels (
                event_id VARCHAR,
                label VARCHAR,
                is_attack INTEGER,
                day VARCHAR
            )
        """)
        connection.execute("""
            INSERT INTO events VALUES
                ('wed-benign', '2017-07-05 10:00:00', '192.0.2.10', '198.51.100.1', 443),
                ('thu-attack', '2017-07-06 10:00:00', '192.0.2.10', '198.51.100.1', 4444),
                ('thu-benign', '2017-07-06 10:02:00', '192.0.2.10', '198.51.100.1', 443),
                ('thu-missed', '2017-07-06 10:10:00', '192.0.2.11', '198.51.100.2', 443),
                ('fri-attack', '2017-07-07 10:00:00', '192.0.2.12', '198.51.100.3', 4444)
        """)
        connection.execute("""
            INSERT INTO flow_features VALUES
                ('wed-benign', 1200, 443, 0.0, 1.0),
                ('thu-attack', 1201, 4444, 1.0, 0.0),
                ('thu-benign', 1202, 443, 0.45, 0.55),
                ('thu-missed', 1203, 443, 0.2, 0.8),
                ('fri-attack', 1204, 4444, 1.0, 0.0)
        """)
        connection.execute("""
            INSERT INTO labels VALUES
                ('wed-benign', 'BENIGN', 0, 'Wednesday'),
                ('thu-attack', 'DDoS', 1, 'Thursday'),
                ('thu-benign', 'BENIGN', 0, 'Thursday'),
                ('thu-missed', 'Bot', 1, 'Thursday'),
                ('fri-attack', 'DDoS', 1, 'Friday')
        """)

    training_features = pd.DataFrame({
        "feature_a": [0.0, 0.1, 0.9, 1.0],
        "feature_b": [1.0, 0.9, 0.1, 0.0],
    })
    model = LogisticRegression(random_state=0).fit(
        training_features, [0, 0, 1, 1]
    )
    artifact_path = tmp_path / "binary.joblib"
    save_model_artifact(
        model=model,
        feature_names=["feature_a", "feature_b"],
        threshold=0.48,
        target_fpr=0.01,
        path=artifact_path,
        split="unseen",
        seed=42,
    )

    events = load_event_features(database)
    assert "label" not in events.columns
    scored = score_events(events, load_model_artifact(artifact_path))
    scored_path = tmp_path / "scored.parquet"
    write_scored_events(scored, scored_path)
    persisted_scores = pd.read_parquet(scored_path)
    ground_truth = load_test_ground_truth(database, ["Thursday", "Friday"])

    evaluated, event_metrics, _, incidents, incident_metrics, _ = (
        evaluate_scored_events(
            persisted_scores,
            ground_truth,
            {"BENIGN": "Benign", "DDoS": "DoS", "Bot": "Bot"},
            test_days=["Thursday", "Friday"],
            entity_columns=["src_ip", "dst_ip"],
            window_minutes=30,
            max_gap_minutes=5,
        )
    )

    assert set(evaluated["event_id"]) == {
        "thu-attack", "thu-benign", "thu-missed", "fri-attack"
    }
    assert event_metrics["test_events"] == 4
    assert event_metrics["true_positives"] == 2
    assert event_metrics["false_positives"] == 1
    assert event_metrics["false_negatives"] == 1
    assert incident_metrics["incident_count"] == 2
    assert incident_metrics["mixed_attack_benign_incidents"] == 1
    assert set(incidents["event_ids"].explode()) == {
        "thu-attack", "thu-benign", "fri-attack"
    }
