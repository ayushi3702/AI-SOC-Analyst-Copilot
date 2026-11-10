import duckdb
import pytest

from agents.lanl_source import LanlFeatureHunter, LoginIncidentContext


@pytest.fixture
def feature_parquet(tmp_path):
    parquet = tmp_path / "lanl_features.parquet"
    database = tmp_path / "features.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("""
            CREATE TABLE features (
                user_id VARCHAR, hour_index BIGINT, event_count BIGINT,
                success_count BIGINT, failure_count BIGINT,
                previous_24h_events BIGINT, previous_24h_failures BIGINT,
                previous_24h_successes BIGINT,
                hours_since_previous_activity DOUBLE,
                redteam_event_count BIGINT
            )
        """)
        connection.execute("""
            INSERT INTO features VALUES
                ('U1', 10, 4, 3, 1, 8, 2, 6, 1.0, 0),
                ('U1', 11, 5, 5, 0, 9, 1, 8, 1.0, 1),
                ('U2', 11, 8, 8, 0, 0, 0, 0, NULL, 0)
        """)
        connection.execute(
            "COPY features TO ? (FORMAT PARQUET)",
            [str(parquet)],
        )
    return parquet


def _context(user_id="U1"):
    return LoginIncidentContext(
        incident_id="lanl-incident-1",
        user_id=user_id,
        start_hour=10,
        end_hour=11,
    )


def test_search_returns_citable_user_hour_rows(feature_parquet):
    records = LanlFeatureHunter(feature_parquet).search(_context())

    assert [record.evidence_id for record in records] == [
        "lanl:U1:10",
        "lanl:U1:11",
    ]
    assert records[0].failure_count == 1
    assert not hasattr(records[1], "redteam_event_count")


def test_search_uses_exact_user_id_and_bound_values(feature_parquet):
    hunter = LanlFeatureHunter(feature_parquet)
    assert hunter.search(_context("U2' OR 1=1 --")) == []


def test_search_caps_and_validates_result_limit(feature_parquet):
    hunter = LanlFeatureHunter(feature_parquet)
    assert len(hunter.search(_context(), limit=1)) == 1
    with pytest.raises(ValueError, match="limit"):
        hunter.search(_context(), limit=501)
    with pytest.raises(TypeError, match="integer"):
        hunter.search(_context(), limit=True)


def test_login_context_requires_valid_hour_range():
    with pytest.raises(ValueError, match="start_hour"):
        LoginIncidentContext("i1", "U1", 3, 2)
    with pytest.raises(ValueError, match="non-negative"):
        LoginIncidentContext("i1", "U1", -1, 2)


def test_missing_feature_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_login_data"):
        LanlFeatureHunter(tmp_path / "missing.parquet")
