import sqlite3

import pytest

from agents.response_draft import (
    DryRunDecisionStore,
    ResponseProposal,
    draft_response_proposals,
)


def test_response_drafts_are_cited_and_only_manual_review():
    evidence = [{"evidence_id": "event:e1"}, {"evidence_id": "event:e2"}]
    mapping = {
        "status": "hypotheses_found",
        "hypotheses": [{"evidence_ids": ["event:e1", "event:e2"]}],
    }

    proposals = draft_response_proposals("i1", evidence, mapping)

    assert [proposal.action for proposal in proposals] == [
        "review_and_preserve_network_evidence",
        "analyst_review_of_port_fanout",
        "simulate_block_observed_source_ip",
    ]
    assert all(proposal.status == "pending_approval" for proposal in proposals)
    assert all(proposal.execution_mode == "dry_run_only" for proposal in proposals)
    assert proposals[0].evidence_ids == ["event:e1", "event:e2"]


def test_response_store_records_approval_without_execution(tmp_path):
    proposal = ResponseProposal(
        proposal_id="proposal-1",
        incident_id="i1",
        action="review_and_preserve_network_evidence",
        rationale="Review evidence.",
        evidence_ids=["event:e1"],
    )
    store = DryRunDecisionStore(tmp_path / "decisions.sqlite3")

    decision = store.record_decision(proposal, "approve", "analyst-1", "Reviewed")

    assert decision["decision"] == "approve"
    assert decision["execution_mode"] == "dry_run_only"
    assert decision["executed"] is False
    assert store.list_decisions("i1") == [decision]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT executed FROM response_decisions"
        ).fetchone() == (0,)
    with pytest.raises(ValueError, match="already has"):
        store.record_decision(proposal, "reject", "analyst-2", "Second decision")


@pytest.mark.parametrize(
    "decision,analyst,reason",
    [
        ("execute", "analyst", "because"),
        ("approve", "", "because"),
        ("reject", "analyst", ""),
    ],
)
def test_response_store_rejects_invalid_decisions(decision, analyst, reason, tmp_path):
    proposal = ResponseProposal("p1", "i1", "review", "why", ["event:e1"])
    store = DryRunDecisionStore(tmp_path / "decisions.sqlite3")

    with pytest.raises(ValueError):
        store.record_decision(proposal, decision, analyst, reason)
