from dataclasses import replace
from datetime import datetime

import pytest

from agents.attack_mapping import (
    ATTACK_ENTERPRISE_URL,
    MIN_DISTINCT_DESTINATION_PORTS,
    TRUNCATED_EVIDENCE_NOTICE,
    map_attack_hypotheses,
)
from agents.log_hunter import EvidenceRecord, IncidentContext


@pytest.fixture
def incident():
    return IncidentContext(
        incident_id="incident-1",
        entity="192.0.2.1",
        start_time=datetime(2026, 1, 2, 9),
        end_time=datetime(2026, 1, 2, 10),
    )


def _record(event_number, source, target, destination_port, event_type="network_flow"):
    return EvidenceRecord(
        evidence_id=f"event:e{event_number}",
        event_id=f"e{event_number}",
        observed_at="2026-01-02 09:00:00",
        source_dataset="test",
        event_type=event_type,
        src_ip=source,
        src_port=40000 + event_number,
        dst_ip=target,
        dst_port=destination_port,
        protocol="tcp",
        duration_us=10,
        fwd_packets=1,
        bwd_packets=1,
        fwd_bytes=100.0,
        bwd_bytes=100.0,
    )


def test_maps_only_matching_source_to_same_target_port_fanout(incident):
    evidence = [
        _record(index, "192.0.2.1", "198.51.100.10", 80 + index)
        for index in range(MIN_DISTINCT_DESTINATION_PORTS)
    ]

    result = map_attack_hypotheses(
        incident, evidence, possibly_truncated=False
    )

    assert result.status == "hypotheses_found"
    assert len(result.hypotheses) == 1
    hypothesis = result.hypotheses[0]
    assert hypothesis.technique_id == "T1046"
    assert hypothesis.technique_name == "Network Service Discovery"
    assert hypothesis.tactic == "Discovery"
    assert hypothesis.confidence == "low"
    assert hypothesis.observed_target == "198.51.100.10"
    assert hypothesis.distinct_destination_ports == MIN_DISTINCT_DESTINATION_PORTS
    assert hypothesis.reference == ATTACK_ENTERPRISE_URL
    assert len(hypothesis.evidence_ids) == MIN_DISTINCT_DESTINATION_PORTS
    assert "Hypothesis only" in hypothesis.assessment


def test_incoming_flows_do_not_count_as_source_discovery(incident):
    evidence = [
        _record(index, "198.51.100.10", "192.0.2.1", 80 + index)
        for index in range(MIN_DISTINCT_DESTINATION_PORTS)
    ]

    result = map_attack_hypotheses(
        incident, evidence, possibly_truncated=False
    )

    assert result.status == "no_pattern_matched"
    assert result.hypotheses == []


def test_distinct_ports_must_share_a_single_target(incident):
    evidence = [
        _record(index, "192.0.2.1", f"198.51.100.{index}", 80 + index)
        for index in range(MIN_DISTINCT_DESTINATION_PORTS)
    ]

    result = map_attack_hypotheses(
        incident, evidence, possibly_truncated=False
    )

    assert result.status == "no_pattern_matched"


def test_result_is_not_evaluated_when_events_may_be_truncated(incident):
    evidence = [
        _record(index, "192.0.2.1", "198.51.100.10", 80 + index)
        for index in range(MIN_DISTINCT_DESTINATION_PORTS)
    ]

    result = map_attack_hypotheses(
        incident, evidence, possibly_truncated=True
    )

    assert result.status == "not_evaluated"
    assert result.notice == TRUNCATED_EVIDENCE_NOTICE
    assert result.hypotheses == []


def test_missing_fields_or_non_network_records_stop_evaluation(incident):
    incomplete = replace(
        _record(1, "192.0.2.1", "198.51.100.10", 80),
        dst_port=None,
    )
    non_network = replace(
        _record(2, "192.0.2.1", "198.51.100.10", 81),
        event_type="endpoint_event",
    )

    for record in (incomplete, non_network):
        result = map_attack_hypotheses(
            incident, [record], possibly_truncated=False
        )
        assert result.status == "not_evaluated"
        assert result.hypotheses == []


@pytest.mark.parametrize("threshold", [1, 1.5, True])
def test_invalid_port_threshold_is_rejected(incident, threshold):
    with pytest.raises(ValueError, match="integer"):
        map_attack_hypotheses(
            incident,
            [],
            possibly_truncated=False,
            minimum_distinct_destination_ports=threshold,
        )
