from datetime import datetime, timezone

import duckdb
import pytest

from agents.log_hunter import (
    DuckDBLogHunter,
    EventSearchFilters,
    IncidentContext,
)


@pytest.fixture
def event_database(tmp_path):
    database = tmp_path / "events.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("""
            CREATE TABLE events (
                event_id VARCHAR, timestamp TIMESTAMP, source_dataset VARCHAR,
                event_type VARCHAR, src_ip VARCHAR, src_port BIGINT,
                dst_ip VARCHAR, dst_port BIGINT, protocol VARCHAR,
                duration_us BIGINT, fwd_packets BIGINT, bwd_packets BIGINT,
                fwd_bytes DOUBLE, bwd_bytes DOUBLE
            )
        """)
        connection.execute("""
            INSERT INTO events VALUES
                ('e2', '2026-01-02 10:00:00', 'test', 'network_flow',
                 '192.0.2.1', 1234, '198.51.100.2', 443, 'tcp',
                 20, 2, 1, 200, 100),
                ('e1', '2026-01-02 09:00:00', 'test', 'network_flow',
                 '198.51.100.1', 443, '192.0.2.1', 1235, 'tcp',
                 10, 1, 1, 100, 100),
                ('outside', '2026-01-02 10:00:01', 'test', 'network_flow',
                 '192.0.2.1', 1234, '198.51.100.2', 443, 'tcp',
                 20, 2, 1, 200, 100)
        """)
        connection.execute("""
            CREATE TABLE labels (event_id VARCHAR, label VARCHAR)
        """)
        connection.execute("""
            INSERT INTO labels VALUES ('e1', 'must-not-be-exposed')
        """)
    return database


def _incident(entity="192.0.2.1"):
    return IncidentContext(
        incident_id="incident-1",
        entity=entity,
        start_time=datetime(2026, 1, 2, 9),
        end_time=datetime(2026, 1, 2, 10),
    )


def test_search_returns_ordered_citable_events_and_never_labels(event_database):
    evidence = DuckDBLogHunter(event_database).search(_incident())

    assert [record.event_id for record in evidence] == ["e1", "e2"]
    assert evidence[0].evidence_id == "event:e1"
    assert evidence[0].observed_at == "2026-01-02 09:00:00"
    assert evidence[0].src_ip == "198.51.100.1"
    assert evidence[0].dst_ip == "192.0.2.1"
    assert not hasattr(evidence[0], "label")


def test_search_binds_entity_as_data(event_database):
    evidence = DuckDBLogHunter(event_database).search(
        _incident("192.0.2.1' OR 1=1 --")
    )

    assert evidence == []


def test_search_applies_result_limit(event_database):
    evidence = DuckDBLogHunter(event_database).search(_incident(), limit=1)

    assert [record.event_id for record in evidence] == ["e1"]


def test_search_supports_parameterized_fixed_field_filters(event_database):
    evidence = DuckDBLogHunter(event_database).search(
        _incident(),
        filters=EventSearchFilters(
            src_ip="198.51.100.1",
            destination_port=1235,
            protocol="tcp",
        ),
    )

    assert [record.event_id for record in evidence] == ["e1"]


@pytest.mark.parametrize("limit", [0, 501])
def test_search_rejects_out_of_range_limit(event_database, limit):
    with pytest.raises(ValueError, match="limit"):
        DuckDBLogHunter(event_database).search(_incident(), limit=limit)


def test_search_rejects_non_integer_limit(event_database):
    with pytest.raises(TypeError, match="integer"):
        DuckDBLogHunter(event_database).search(_incident(), limit=1.5)


def test_incident_context_rejects_invalid_time_range_and_timezone():
    with pytest.raises(ValueError, match="start_time"):
        IncidentContext(
            "incident-1",
            "192.0.2.1",
            datetime(2026, 1, 2, 11),
            datetime(2026, 1, 2, 10),
        )

    with pytest.raises(ValueError, match="timezone-naive"):
        IncidentContext(
            "incident-1",
            "192.0.2.1",
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            datetime(2026, 1, 3),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_port": -1},
        {"destination_port": 65536},
        {"source_port": True},
        {"protocol": ""},
        {"src_ip": " "},
    ],
)
def test_event_search_filters_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        EventSearchFilters(**kwargs)
