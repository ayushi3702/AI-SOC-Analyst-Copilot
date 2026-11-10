"""Cautious, deterministic ATT&CK hypotheses from bounded network-flow evidence."""

from dataclasses import asdict, dataclass

from agents.log_hunter import EvidenceRecord, IncidentContext


MIN_DISTINCT_DESTINATION_PORTS = 10
ATTACK_ENTERPRISE_URL = (
    "https://attack.mitre.org/techniques/T1046/"
)
TRUNCATED_EVIDENCE_NOTICE = (
    "ATT&CK hypotheses were not evaluated because the event results may be "
    "truncated. Increase the query limit or narrow the time window."
)
NO_MATCH_NOTICE = (
    "No configured ATT&CK behavior pattern met its documented threshold in "
    "the retrieved evidence. This does not rule out ATT&CK techniques."
)
INSUFFICIENT_FIELDS_NOTICE = (
    "ATT&CK hypotheses were not evaluated because the evidence set is missing "
    "required source, destination, port, or event-type fields."
)


@dataclass(frozen=True)
class AttackHypothesis:
    technique_id: str
    technique_name: str
    tactic: str
    assessment: str
    confidence: str
    observed_source: str
    observed_target: str
    distinct_destination_ports: int
    evidence_ids: list[str]
    reference: str


@dataclass(frozen=True)
class AttackMappingResult:
    status: str
    notice: str | None
    hypotheses: list[AttackHypothesis]


def map_attack_hypotheses(
    incident: IncidentContext,
    evidence: list[EvidenceRecord],
    *,
    possibly_truncated: bool,
    minimum_distinct_destination_ports: int = MIN_DISTINCT_DESTINATION_PORTS,
) -> AttackMappingResult:
    """Suggest T1046 only for a complete, exact port-fanout pattern.

    The threshold is a transparent project heuristic, not a MITRE requirement
    or a validated detection rule. Candidates are hypotheses, never confirmed
    techniques.
    """
    if not isinstance(incident, IncidentContext):
        raise TypeError("incident must be an IncidentContext")
    if not isinstance(possibly_truncated, bool):
        raise TypeError("possibly_truncated must be a boolean")
    if (
        isinstance(minimum_distinct_destination_ports, bool)
        or not isinstance(minimum_distinct_destination_ports, int)
        or minimum_distinct_destination_ports < 2
    ):
        raise ValueError("minimum_distinct_destination_ports must be an integer >= 2")
    if possibly_truncated:
        return AttackMappingResult(
            status="not_evaluated",
            notice=TRUNCATED_EVIDENCE_NOTICE,
            hypotheses=[],
        )
    if any(
        not isinstance(record, EvidenceRecord)
        or not record.src_ip
        or not record.dst_ip
        or record.dst_port is None
        or record.event_type != "network_flow"
        for record in evidence
    ):
        return AttackMappingResult(
            status="not_evaluated",
            notice=INSUFFICIENT_FIELDS_NOTICE,
            hypotheses=[],
        )

    grouped: dict[tuple[str, str], list[EvidenceRecord]] = {}
    for record in evidence:
        if record.src_ip == incident.entity:
            grouped.setdefault((record.src_ip, record.dst_ip), []).append(record)

    hypotheses = []
    for (source, target), records in sorted(grouped.items()):
        ports = {record.dst_port for record in records}
        if len(ports) < minimum_distinct_destination_ports:
            continue
        hypotheses.append(
            AttackHypothesis(
                technique_id="T1046",
                technique_name="Network Service Discovery",
                tactic="Discovery",
                assessment=(
                    "Hypothesis only: this source contacted one destination "
                    "on many distinct destination ports. Flow data alone "
                    "does not establish intent or confirm service discovery."
                ),
                confidence="low",
                observed_source=source,
                observed_target=target,
                distinct_destination_ports=len(ports),
                evidence_ids=sorted(
                    record.evidence_id for record in records
                ),
                reference=ATTACK_ENTERPRISE_URL,
            )
        )
    if not hypotheses:
        return AttackMappingResult(
            status="no_pattern_matched",
            notice=NO_MATCH_NOTICE,
            hypotheses=[],
        )
    return AttackMappingResult(
        status="hypotheses_found",
        notice=None,
        hypotheses=hypotheses,
    )


def serialize_mapping_result(result: AttackMappingResult) -> dict:
    return {
        "status": result.status,
        "notice": result.notice,
        "hypotheses": [asdict(hypothesis) for hypothesis in result.hypotheses],
    }
