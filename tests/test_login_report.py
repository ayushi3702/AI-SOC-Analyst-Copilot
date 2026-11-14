from agents.evidence_report import UnsupportedClaimError
from agents.lanl_source import LoginEvidenceRecord
from agents.login_report import (
    LOGIN_AUTHOR_PROMPT,
    generate_login_report,
)


class FakeClient:
    model = "test-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, prompt, payload):
        self.calls.append((prompt, payload))
        return self.responses.pop(0)


def _record():
    return LoginEvidenceRecord(
        evidence_id="lanl:U1:12",
        user_id="U1",
        hour_index=12,
        event_count=4,
        success_count=3,
        failure_count=1,
        previous_24h_events=10,
        previous_24h_failures=2,
        previous_24h_successes=8,
        hours_since_previous_activity=1.0,
    )


def _client(verdict=True, evidence_id="lanl:U1:12"):
    return FakeClient([
        {
            "summary": "Four authentication events occurred in relative hour 12.",
            "summary_evidence_ids": [evidence_id],
            "claims": [{
                "statement": "One authentication event failed.",
                "evidence_ids": [evidence_id],
            }],
        },
        {
            "summary_supported": verdict,
            "claims_supported": [verdict],
        },
    ])


def test_login_report_is_cited_and_verified_without_labels():
    record = _record()
    client = _client()
    incident = {
        "incident_id": "i1",
        "user_id": "U1",
        "start_hour": 12,
        "end_hour": 12,
    }

    report = generate_login_report(incident, [record], 10, client)

    assert report["verification"]["verified"] is True
    assert report["claims"][0]["evidence_ids"] == ["lanl:U1:12"]
    assert "relative hour" in report["time_limitation"]
    assert "redteam_event_count" not in report["evidence"][0]
    assert client.calls[0][0] == LOGIN_AUTHOR_PROMPT


def test_login_report_fails_closed_for_verifier_rejection():
    client = _client(verdict=False)

    try:
        generate_login_report({"incident_id": "i1"}, [_record()], 10, client)
    except UnsupportedClaimError as error:
        assert "rejected" in str(error)
    else:
        raise AssertionError("Unsupported LANL claim unexpectedly passed")


def test_login_report_rejects_unknown_evidence_citation():
    client = _client(evidence_id="lanl:unknown:0")

    try:
        generate_login_report({"incident_id": "i1"}, [_record()], 10, client)
    except UnsupportedClaimError as error:
        assert "citations" in str(error)
    else:
        raise AssertionError("Unknown evidence citation unexpectedly passed")


def test_login_report_rejects_non_string_evidence_citation():
    client = _client(evidence_id={"unexpected": "object"})

    try:
        generate_login_report({"incident_id": "i1"}, [_record()], 10, client)
    except UnsupportedClaimError as error:
        assert "citations" in str(error)
    else:
        raise AssertionError("Malformed evidence citation unexpectedly passed")


def test_empty_login_report_does_not_call_model():
    client = FakeClient([])
    report = generate_login_report({"incident_id": "i1"}, [], 10, client)

    assert report["status"] == "insufficient_evidence"
    assert client.calls == []
