import numpy as np
import pandas as pd
import pytest

from ml.train_login_anomaly import (
    LOGIN_FEATURES,
    score_login_features,
    train_login_anomaly_detector,
    train_login_anomaly_from_parquet,
    transform_login_features,
)


def _login_frame():
    rng = np.random.default_rng(42)
    rows = []
    for hour in range(6):
        for user_number in range(40):
            row = {
                "user_id": f"user-{user_number:02d}",
                "hour_index": hour,
                "redteam_event_count": 0,
            }
            row.update({
                name: float(rng.poisson(2))
                for name in LOGIN_FEATURES[:-2]
            })
            row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[
        (frame["hour_index"] >= 2)
        & (frame["hour_index"] < 4)
        & (frame["user_id"] == "user-00"),
        "redteam_event_count",
    ] = 1
    test_attacks = (
        (frame["hour_index"] >= 4) & frame["user_id"].isin(["user-00", "user-01"])
    )
    frame.loc[test_attacks, "redteam_event_count"] = 1
    frame.loc[test_attacks, "event_count"] = 50
    frame.loc[test_attacks, "failure_count"] = 20
    return frame


def test_login_detector_uses_chronological_split_and_fixed_validation_threshold():
    report, artifact = train_login_anomaly_detector(
        _login_frame(),
        fit_end_hour=2,
        validation_end_hour=4,
        max_train_rows=50,
        n_estimators=20,
        seed=7,
    )

    assert artifact["task"] == "login_anomaly"
    assert report["method"]["split"].startswith("chronological")
    assert report["method"]["fit_redteam_rows_excluded"]
    assert report["method"]["training_rows_used"] == 50
    assert report["method"]["validation_redteam_user_hours"] == 2
    assert report["method"]["test_redteam_user_hours"] == 4
    assert report["validation_metrics"]["false_positive_rate"] <= 0.05
    assert report["test_metrics"]["rows"] == 80
    scored = score_login_features(_login_frame().tail(80), artifact)
    assert scored["anomaly_score"].notna().all()
    assert scored["is_alert"].isin([True, False]).all()


def test_login_detector_rejects_test_period_without_redteam_labels():
    frame = _login_frame()
    frame.loc[frame["hour_index"] >= 4, "redteam_event_count"] = 0

    with pytest.raises(ValueError, match="red-team"):
        train_login_anomaly_detector(
            frame,
            fit_end_hour=2,
            validation_end_hour=4,
            max_train_rows=50,
            n_estimators=10,
        )


def test_login_detector_is_invariant_to_input_row_order():
    frame = _login_frame()
    options = {
        "fit_end_hour": 2,
        "validation_end_hour": 4,
        "max_train_rows": 50,
        "n_estimators": 20,
        "seed": 7,
    }

    original, _ = train_login_anomaly_detector(frame, **options)
    shuffled, _ = train_login_anomaly_detector(
        frame.sample(frac=1, random_state=17), **options
    )

    assert original["validation_metrics"] == shuffled["validation_metrics"]
    assert original["test_metrics"] == shuffled["test_metrics"]


def test_parquet_login_training_scores_validation_and_test_in_batches(tmp_path):
    frame = _login_frame()
    input_path = tmp_path / "login_features.parquet"
    frame.to_parquet(input_path, index=False)

    report, artifact = train_login_anomaly_from_parquet(
        input_path,
        fit_end_hour=2,
        validation_end_hour=4,
        max_train_rows=50,
        n_estimators=20,
        batch_size=13,
        seed=7,
    )

    assert artifact["task"] == "login_anomaly"
    assert report["method"]["training_rows_used"] == 50
    assert report["method"]["fit_rows_in_period"] == 80
    assert report["method"]["validation_rows_in_period"] == 80
    assert report["method"]["test_rows_in_period"] == 80
    assert report["method"]["evaluation_batch_size"] == 13
    assert report["test_metrics"]["rows"] == 80


def test_login_feature_transform_rejects_negative_counts():
    frame = _login_frame().head(1)
    frame.loc[:, "failure_count"] = -1

    with pytest.raises(ValueError, match="non-negative"):
        transform_login_features(frame)


def test_login_feature_transform_log_scales_counts_and_time_once():
    frame = _login_frame().head(1).copy()
    frame.loc[:, "event_count"] = 3
    frame.loc[:, "hours_since_previous_activity"] = 3

    transformed = transform_login_features(frame)

    assert transformed.loc[0, "event_count"] == pytest.approx(np.log1p(3))
    assert transformed.loc[
        0, "hours_since_previous_activity"
    ] == pytest.approx(np.log1p(3))


def test_login_scoring_rejects_artifact_with_wrong_feature_schema():
    frame = _login_frame()
    _, artifact = train_login_anomaly_detector(
        frame,
        fit_end_hour=2,
        validation_end_hour=4,
        max_train_rows=50,
        n_estimators=10,
    )
    artifact["feature_names"] = ["wrong"]

    with pytest.raises(ValueError, match="feature schema"):
        score_login_features(frame.tail(10), artifact)
