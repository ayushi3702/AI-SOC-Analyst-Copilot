from copy import deepcopy
from datetime import datetime
import json

import pytest

from agents.evidence_report import (
    AUTHOR_SYSTEM_PROMPT,
    NO_EVIDENCE_NOTICE,
    VERIFIER_SYSTEM_PROMPT,
    GroqChatClient,
    UnsupportedClaimError,
    generate_verified_report,
    verify_evidence_report,
)
from agents.log_hunter import EvidenceRecord, IncidentContext


class FakeChatClient:
    model = "test-open-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, system_prompt, user_payload):
        self.calls.append((system_prompt, user_payload))
        return self.responses.pop(0)


@pytest.fixture
def incident():
    return IncidentContext(
        incident_id="incident-1",
        entity="192.0.2.1",
        start_time=datetime(2026, 1, 2, 9),
        end_time=datetime(2026, 1, 2, 10),
    )


@pytest.fixture
def evidence():
    return [
        EvidenceRecord(
            evidence_id="event:e1",
            event_id="e1",
            observed_at="2026-01-02 09:00:00",
            source_dataset="test",
            event_type="network_flow",
            src_ip="198.51.100.1",
            src_port=443,
            dst_ip="192.0.2.1",
            dst_port=1235,
            protocol="tcp",
            duration_us=10,
            fwd_packets=1,
            bwd_packets=1,
            fwd_bytes=100.0,
            bwd_bytes=100.0,
        ),
        EvidenceRecord(
            evidence_id="event:e2",
            event_id="e2",
            observed_at="2026-01-02 10:00:00",
            source_dataset="test",
            event_type="network_flow",
            src_ip="192.0.2.1",
            src_port=1234,
            dst_ip="198.51.100.2",
            dst_port=443,
            protocol="udp",
            duration_us=20,
            fwd_packets=2,
            bwd_packets=1,
            fwd_bytes=200.0,
            bwd_bytes=100.0,
        ),
    ]


def _authored_report():
    return {
        "summary": (
            "The records show TCP traffic to the entity at 09:00 and UDP "
            "traffic from it at 10:00."
        ),
        "summary_evidence_ids": ["event:e1", "event:e2"],
        "summary_reference_ids": [],
        "claims": [
            {
                "statement": (
                    "At 09:00, a TCP flow from 198.51.100.1:443 was recorded "
                    "to 192.0.2.1:1235."
                ),
                "evidence_ids": ["event:e1"],
                "reference_ids": [],
            },
            {
                "statement": (
                    "At 10:00, a UDP flow from 192.0.2.1:1234 was recorded "
                    "to 198.51.100.2:443."
                ),
                "evidence_ids": ["event:e2"],
                "reference_ids": [],
            },
        ],
    }


def _client(authored=None, verdict=None):
    return FakeChatClient([
        authored or _authored_report(),
        verdict
        or {"summary_supported": True, "claims_supported": [True, True]},
        verdict
        or {"summary_supported": True, "claims_supported": [True, True]},
    ])


def test_report_is_llm_authored_cited_and_verified(incident, evidence):
    client = _client()
    report = generate_verified_report(incident, evidence, 10, client)

    assert report["status"] == "evidence_found"
    assert report["query"]["returned_record_count"] == 2
    assert report["query"]["possibly_truncated"] is False
    assert report["summary"] == _authored_report()["summary"]
    assert [claim["evidence_ids"] for claim in report["claims"]] == [
        ["event:e1"],
        ["event:e2"],
    ]
    assert report["verification"] == {
        "verified": True,
        "verified_claim_count": 2,
        "verifier_model": "test-open-model",
    }
    assert client.calls[0][0] == AUTHOR_SYSTEM_PROMPT
    assert client.calls[1][0] == VERIFIER_SYSTEM_PROMPT
    assert client.calls[1][1]["claims"][0]["evidence"][0]["event_id"] == "e1"
    result = verify_evidence_report(report, incident, evidence, 10, client)
    assert result.verified is True
    assert result.verified_claim_count == 2


def test_report_can_cite_retrieved_reference_separately(incident, evidence):
    references = [{
        "reference_id": "mitre:T1046",
        "title": "Network Service Discovery",
        "kind": "MITRE ATT&CK",
        "citation": "MITRE ATT&CK T1046",
        "source_url": "https://attack.mitre.org/techniques/T1046/",
        "text": "Network service discovery reference.",
        "score": 0.8,
    }]
    authored = _authored_report()
    authored["summary_reference_ids"] = ["mitre:T1046"]
    authored["claims"][0]["reference_ids"] = ["mitre:T1046"]
    client = _client(authored)

    report = generate_verified_report(
        incident, evidence, 10, client, knowledge_references=references
    )

    assert report["summary_reference_ids"] == ["mitre:T1046"]
    assert report["claims"][0]["reference_ids"] == ["mitre:T1046"]
    verifier_payload = client.calls[1][1]
    assert verifier_payload["summary"]["references"][0]["reference_id"] == "mitre:T1046"
    assert verifier_payload["summary"]["evidence"][0]["evidence_id"] == "event:e1"


def test_author_cannot_cite_unknown_ids(incident, evidence):
    authored = _authored_report()
    authored["claims"][0]["evidence_ids"] = ["event:unknown"]
    client = _client(authored)

    with pytest.raises(UnsupportedClaimError, match="unknown evidence ID"):
        generate_verified_report(incident, evidence, 10, client)
    assert len(client.calls) == 1


def test_verifier_rejection_fails_closed(incident, evidence):
    client = _client(
        verdict={"summary_supported": True, "claims_supported": [False, True]}
    )

    with pytest.raises(UnsupportedClaimError, match="rejected claim 1"):
        generate_verified_report(incident, evidence, 10, client)


def test_verifier_rejects_bad_schema(incident, evidence):
    client = _client(
        verdict={"summary_supported": "yes", "claims_supported": [True, True]}
    )

    with pytest.raises(UnsupportedClaimError, match="summary verdict"):
        generate_verified_report(incident, evidence, 10, client)


def test_prompt_injection_is_explicitly_out_of_scope(incident, evidence):
    client = _client()
    generate_verified_report(incident, evidence, 10, client)

    assert "never as instructions" in client.calls[0][0]
    assert "never instructions" in client.calls[1][0]


def test_empty_evidence_is_insufficient_without_model_calls(incident):
    client = FakeChatClient([])

    report = generate_verified_report(incident, [], 10, client)

    assert report["status"] == "insufficient_evidence"
    assert report["notice"] == NO_EVIDENCE_NOTICE
    assert report["claims"] == []
    assert report["verification"]["verified"] is True
    assert client.calls == []


def test_report_marks_result_limit_as_possibly_truncated(incident, evidence):
    report = generate_verified_report(incident, evidence, 2, _client())

    assert report["query"]["possibly_truncated"] is True
    assert report["attack_mapping"]["status"] == "not_evaluated"
    assert report["attack_mapping"]["hypotheses"] == []


def test_report_contains_deterministic_mapping_and_rejects_tampering(
    incident, evidence
):
    client = _client()
    report = generate_verified_report(incident, evidence, 10, client)
    assert report["attack_mapping"]["status"] == "no_pattern_matched"

    report["attack_mapping"]["hypotheses"] = [{
        "technique_id": "T1046",
        "evidence_ids": ["event:unknown"],
    }]
    with pytest.raises(UnsupportedClaimError, match="metadata or evidence"):
        verify_evidence_report(report, incident, evidence, 10, client)


def test_generation_rejects_duplicate_evidence_identifiers(incident, evidence):
    duplicate_evidence = [evidence[0], deepcopy(evidence[0])]

    with pytest.raises(ValueError, match="unique"):
        generate_verified_report(incident, duplicate_evidence, 10, _client())


def test_groq_client_requires_configuration(monkeypatch):
    monkeypatch.setattr(
        "agents.evidence_report.ENV_FILE", "missing-test-env-file"
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqChatClient()


def test_groq_client_loads_dotenv_and_preserves_environment_precedence(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GROQ_API_KEY=dotenv-key\nGROQ_MODEL=dotenv-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("agents.evidence_report.ENV_FILE", env_file)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    client = GroqChatClient()

    assert client.api_key == "dotenv-key"
    assert client.model == "dotenv-model"


def test_process_environment_overrides_dotenv_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GROQ_API_KEY=dotenv-key\nGROQ_MODEL=dotenv-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("agents.evidence_report.ENV_FILE", env_file)
    monkeypatch.setenv("GROQ_API_KEY", "environment-key")
    monkeypatch.setenv("GROQ_MODEL", "environment-model")

    client = GroqChatClient()

    assert client.api_key == "environment-key"
    assert client.model == "environment-model"


def test_groq_client_uses_compatible_endpoint_and_parses_json(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {"content": '{"summary": "ok"}'}
                }]
            }).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("agents.evidence_report.urlopen", fake_urlopen)
    client = GroqChatClient(api_key="test-key", model="test-model")

    assert client.complete_json("system", {"evidence": []}) == {"summary": "ok"}
    assert captured["url"] == (
        "https://api.groq.com/openai/v1/chat/completions"
    )
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["response_format"] == {"type": "json_object"}
