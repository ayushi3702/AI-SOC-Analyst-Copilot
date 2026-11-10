"""Evidence-linked response proposals with approval-only dry-run recording."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import uuid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_PATH = ROOT / "data" / "processed" / "response_decisions.sqlite3"


@dataclass(frozen=True)
class ResponseProposal:
    proposal_id: str
    incident_id: str
    action: str
    rationale: str
    evidence_ids: list[str]
    status: str = "pending_approval"
    execution_mode: str = "dry_run_only"


def draft_response_proposals(
    incident_id: str,
    evidence: list[dict],
    attack_mapping: dict,
) -> list[ResponseProposal]:
    """Offer conservative analyst tasks; never invoke a control or external tool."""
    if not isinstance(incident_id, str) or not incident_id.strip():
        raise ValueError("incident_id must be a non-empty string")
    if not isinstance(evidence, list):
        raise TypeError("evidence must be a list")
    proposals = []
    if evidence:
        proposals.append(
            ResponseProposal(
                proposal_id=uuid.uuid4().hex,
                incident_id=incident_id,
                action="review_and_preserve_network_evidence",
                rationale=(
                    "Review the cited flow records and preserve relevant "
                    "telemetry under your organization's process."
                ),
                evidence_ids=[
                    item["evidence_id"]
                    for item in evidence
                    if isinstance(item.get("evidence_id"), str)
                ],
            )
        )
    if attack_mapping.get("status") == "hypotheses_found":
        for hypothesis in attack_mapping.get("hypotheses", []):
            evidence_ids = hypothesis.get("evidence_ids", [])
            if not evidence_ids:
                continue
            proposals.append(
                ResponseProposal(
                    proposal_id=uuid.uuid4().hex,
                    incident_id=incident_id,
                    action="analyst_review_of_port_fanout",
                    rationale=(
                        "Review the low-confidence ATT&CK hypothesis and "
                        "confirm context before considering containment."
                    ),
                    evidence_ids=evidence_ids,
                )
            )
            proposals.append(
                ResponseProposal(
                    proposal_id=uuid.uuid4().hex,
                    incident_id=incident_id,
                    action="simulate_block_observed_source_ip",
                    rationale=(
                        "Optional containment simulation for analyst review "
                        "only. The underlying ATT&CK mapping is low confidence; "
                        "blocking is not executed and could disrupt legitimate "
                        "traffic if applied without validation."
                    ),
                    evidence_ids=evidence_ids,
                )
            )
    return proposals


class DryRunDecisionStore:
    """Persist approval/rejection decisions without executing proposed actions."""

    def __init__(self, path=DEFAULT_AUDIT_PATH):
        self.path = Path(path)

    def record_decision(
        self,
        proposal: ResponseProposal,
        decision: str,
        analyst: str,
        reason: str,
    ) -> dict:
        if not isinstance(proposal, ResponseProposal):
            raise TypeError("proposal must be a ResponseProposal")
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be 'approve' or 'reject'")
        if not isinstance(analyst, str) or not analyst.strip():
            raise ValueError("analyst must be a non-empty identifier")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a non-empty decision reason is required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS response_decisions (
                    decision_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL UNIQUE,
                    incident_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    analyst TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    decided_at TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    executed INTEGER NOT NULL CHECK (executed = 0)
                )
            """)
            connection.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_response_decisions_proposal
                ON response_decisions (proposal_id)
            """)
            row = {
                "decision_id": uuid.uuid4().hex,
                "proposal_id": proposal.proposal_id,
                "incident_id": proposal.incident_id,
                "decision": decision,
                "analyst": analyst.strip(),
                "reason": reason.strip(),
                "decided_at": datetime.now(timezone.utc).isoformat(),
                "execution_mode": "dry_run_only",
                "executed": False,
            }
            try:
                connection.execute(
                    """
                    INSERT INTO response_decisions
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    [
                        row["decision_id"],
                        row["proposal_id"],
                        row["incident_id"],
                        row["decision"],
                        row["analyst"],
                        row["reason"],
                        row["decided_at"],
                        row["execution_mode"],
                    ],
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    "This response proposal already has a recorded decision"
                ) from error
        return row

    def list_decisions(self, incident_id: str) -> list[dict]:
        if not isinstance(incident_id, str) or not incident_id.strip():
            raise ValueError("incident_id must be a non-empty string")
        if not self.path.is_file():
            return []
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT decision_id, proposal_id, incident_id, decision,
                       analyst, reason, decided_at, execution_mode, executed
                FROM response_decisions
                WHERE incident_id = ?
                ORDER BY decided_at
                """,
                [incident_id],
            ).fetchall()
        return [
            {**dict(row), "executed": bool(row["executed"])}
            for row in rows
        ]


def serialize_proposals(proposals: list[ResponseProposal]) -> list[dict]:
    return [asdict(proposal) for proposal in proposals]
