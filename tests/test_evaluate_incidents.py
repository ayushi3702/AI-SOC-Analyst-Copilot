import duckdb
import pandas as pd
import pytest

from ml.evaluate_incidents import evaluate_scored_events, load_test_ground_truth


def _scored_events():
    return pd.DataFrame({
        "event_id": ["wed", "thu-attack", "thu-benign", "thu-missed", "fri-attack"],
        "timestamp": [
            "2017-07-05T23:55:00Z",
            "2017-07-06T00:00:00Z",
            "2017-07-06T00:20:00Z",
            "2017-07-06T00:30:00Z",
            "2017-07-07T00:00:00Z",
        ],
        "src_ip": ["192.0.2.1"] * 4 + ["198.51.100.2"],
        "dst_ip": ["198.51.100.10"] * 5,
        "score": [0.9, 0.9, 0.8, 0.2, 0.95],
        "is_alert": [True, True, True, False, True],
    })


def _test_truth():
    return pd.DataFrame({
        "event_id": ["wed", "thu-attack", "thu-benign", "thu-missed", "fri-attack"],
        "label": ["BENIGN", "DDoS", "BENIGN", "Bot", "Bot"],
        "is_attack": [0, 1, 0, 1, 1],
        "day": ["Wednesday", "Thursday", "Thursday", "Thursday", "Friday"],
    })


def test_evaluator_filters_to_test_days_before_clustering_and_reports_metrics():
    evaluation, events, families, incidents, incident_metrics, samples = (
        evaluate_scored_events(
            _scored_events(),
            _test_truth(),
            {"BENIGN": "Benign", "DDoS": "DoS", "Bot": "Bot"},
            test_days=["Thursday", "Friday"],
            window_minutes=30,
            entity_columns=["src_ip", "dst_ip"],
            max_gap_minutes=5,
        )
    )

    assert set(evaluation["event_id"]) == {
        "thu-attack", "thu-benign", "thu-missed", "fri-attack"
    }
    assert {
        key: value for key, value in events.items()
        if key not in {
            "test_days", "cluster_entity", "cluster_window_minutes",
            "cluster_max_gap_minutes",
        }
    } == {
        "test_events": 4,
        "test_attacks": 3,
        "test_benign": 1,
        "alerts": 3,
        "true_positives": 2,
        "false_positives": 1,
        "false_negatives": 1,
        "recall": pytest.approx(2 / 3),
        "false_positive_rate": 1.0,
        "precision": pytest.approx(2 / 3),
        "alert_rate": 0.75,
    }
    assert events["test_days"] == ["Friday", "Thursday"]
    assert events["cluster_entity"] == ["src_ip", "dst_ip"]
    assert events["cluster_window_minutes"] == 30
    assert events["cluster_max_gap_minutes"] == 5
    assert incidents["event_ids"].tolist() == [
        ["thu-attack"],
        ["thu-benign"],
        ["fri-attack"],
    ]
    assert incidents["attack_count"].tolist() == [1, 0, 1]
    assert incidents["benign_count"].tolist() == [0, 1, 0]
    assert incident_metrics["incident_count"] == 3
    assert incident_metrics["mixed_attack_benign_incidents"] == 0
    assert incident_metrics["attack_alert_events_in_incidents"] == 2
    assert incident_metrics["attack_event_incident_coverage"] == pytest.approx(2 / 3)
    assert set(samples["event_ids"].explode()) == {
        "thu-attack", "thu-benign", "fri-attack"
    }
    assert families.set_index("family")["recall"].to_dict() == {
        "Bot": 0.5,
        "DoS": 1.0,
    }


def test_loader_queries_only_configured_test_days(tmp_path):
    database = tmp_path / "labels.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("""
            CREATE TABLE labels (
                event_id VARCHAR, label VARCHAR, is_attack INTEGER, day VARCHAR
            )
        """)
        connection.execute("""
            INSERT INTO labels VALUES
                ('wed', 'BENIGN', 0, 'Wednesday'),
                ('thu', 'DDoS', 1, 'Thursday'),
                ('fri', 'Bot', 1, 'Friday')
        """)

    truth = load_test_ground_truth(database, ["Thursday", "Friday"])

    assert set(truth["event_id"]) == {"thu", "fri"}


def test_evaluator_rejects_missing_test_predictions_and_bad_input():
    missing_scores = _scored_events().loc[lambda events: events["event_id"] != "fri-attack"]
    with pytest.raises(ValueError, match="missing 1 test event"):
        evaluate_scored_events(
            missing_scores, _test_truth(), {}, test_days=["Thursday", "Friday"]
        )

    duplicate_scores = pd.concat([_scored_events(), _scored_events().iloc[[0]]])
    with pytest.raises(ValueError, match="unique"):
        evaluate_scored_events(
            duplicate_scores, _test_truth(), {}, test_days=["Thursday", "Friday"]
        )
