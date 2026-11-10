import sqlite3

from fastapi.testclient import TestClient
import duckdb

from agents import api


def _network_database(path):
    with duckdb.connect(str(path)) as connection:
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
                ('e1', '2026-01-02 09:00:00', 'test', 'network_flow',
                 '198.51.100.1', 1234, '192.0.2.1', 443, 'tcp',
                 10, 1, 1, 100, 100)
        """)


def _lanl_parquet(path, temporary_database):
    with duckdb.connect(str(temporary_database)) as connection:
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
            INSERT INTO features VALUES ('U1', 12, 4, 3, 1, 8, 2, 6, 1.0, 1)
        """)
        connection.execute(
            "COPY features TO ? (FORMAT PARQUET)", [str(path)]
        )


def test_network_search_and_response_decisions_are_dry_run_only(
    tmp_path, monkeypatch
):
    database = tmp_path / "events.duckdb"
    audit = tmp_path / "audit.sqlite3"
    _network_database(database)
    monkeypatch.setattr(api, "DATABASE_PATH", database)
    monkeypatch.setattr(api, "AUDIT_PATH", audit)
    api.PENDING_PROPOSALS.clear()
    client = TestClient(api.app)

    response = client.post("/api/search/network", json={
        "incident_id": "i1",
        "entity": "192.0.2.1",
        "start_time": "2026-01-02T09:00:00",
        "end_time": "2026-01-02T09:00:00",
        "destination_port": 443,
    })
    assert response.status_code == 200
    result = response.json()
    assert result["evidence_count"] == 1
    assert result["evidence"][0]["event_id"] == "e1"
    assert result["response_proposals"][0]["execution_mode"] == "dry_run_only"
    proposal_id = result["response_proposals"][0]["proposal_id"]

    decision = client.post("/api/responses/decision", json={
        "proposal_id": proposal_id,
        "decision": "approve",
        "analyst": "analyst-1",
        "reason": "Reviewed in test.",
    })
    assert decision.status_code == 200
    assert decision.json()["executed"] is False
    replay = client.post("/api/responses/decision", json={
        "proposal_id": proposal_id,
        "decision": "approve",
        "analyst": "analyst-1",
        "reason": "Replayed decision.",
    })
    assert replay.status_code == 404
    listed = client.get("/api/responses/i1")
    assert listed.status_code == 200
    assert listed.json()["decisions"][0]["executed"] is False
    with sqlite3.connect(audit) as connection:
        assert connection.execute(
            "SELECT executed FROM response_decisions"
        ).fetchone() == (0,)


def test_response_decision_rejects_unknown_or_replayed_proposal():
    client = TestClient(api.app)
    response = client.post("/api/responses/decision", json={
        "proposal_id": "not-found",
        "decision": "approve",
        "analyst": "analyst-1",
        "reason": "Reviewed.",
    })
    assert response.status_code == 404


def test_lanl_search_does_not_expose_redteam_labels(
    tmp_path, monkeypatch
):
    parquet = tmp_path / "lanl.parquet"
    _lanl_parquet(parquet, tmp_path / "source.duckdb")
    monkeypatch.setattr(api, "LANL_FEATURE_PATH", parquet)
    response = TestClient(api.app).post("/api/search/lanl", json={
        "incident_id": "i1",
        "user_id": "U1",
        "start_hour": 12,
        "end_hour": 12,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["evidence"][0]["evidence_id"] == "lanl:U1:12"
    assert "redteam_event_count" not in body["evidence"][0]
    assert body["time_semantics"] == "relative hour index, not wall-clock time"


def test_network_report_uses_references_and_returns_verified_llm_report(
    tmp_path, monkeypatch
):
    database = tmp_path / "events.duckdb"
    _network_database(database)
    monkeypatch.setattr(api, "DATABASE_PATH", database)

    class FakeClient:
        model = "test-model"

        def __init__(self):
            self.responses = [
                {
                    "summary": "A TCP flow to the entity was recorded.",
                    "summary_evidence_ids": ["event:e1"],
                    "summary_reference_ids": [],
                    "claims": [{
                        "statement": "A TCP flow to the entity was recorded.",
                        "evidence_ids": ["event:e1"],
                        "reference_ids": [],
                    }],
                },
                {"summary_supported": True, "claims_supported": [True]},
            ]

        def complete_json(self, system_prompt, payload):
            return self.responses.pop(0)

    monkeypatch.setattr(api, "GroqChatClient", FakeClient)
    response = TestClient(api.app).post("/api/report/network", json={
        "incident_id": "i1",
        "entity": "192.0.2.1",
        "start_time": "2026-01-02T09:00:00",
        "end_time": "2026-01-02T09:00:00",
        "limit": 10,
    })

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["verification"]["verified"] is True
    assert body["report"]["knowledge_references"] == body["knowledge_references"]
    assert body["report"]["claims"][0]["evidence_ids"] == ["event:e1"]


def test_api_rejects_timezone_aware_network_timestamps(
    tmp_path, monkeypatch
):
    database = tmp_path / "events.duckdb"
    _network_database(database)
    monkeypatch.setattr(api, "DATABASE_PATH", database)
    response = TestClient(api.app).post("/api/search/network", json={
        "incident_id": "i1",
        "entity": "192.0.2.1",
        "start_time": "2026-01-02T09:00:00+00:00",
        "end_time": "2026-01-02T09:00:00+00:00",
    })
    assert response.status_code == 400
    assert "timezone-naive" in response.json()["detail"]
