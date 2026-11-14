"""Loopback-only FastAPI endpoints for the local SOC investigation dashboard."""

from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.attack_mapping import map_attack_hypotheses, serialize_mapping_result
from agents.evidence_report import (
    GroqChatClient,
    UnsupportedClaimError,
    generate_verified_report,
)
from agents.knowledge import LocalKnowledgeBase, serialize_references
from agents.login_report import generate_login_report
from agents.lanl_source import (
    DEFAULT_FEATURES,
    LanlFeatureHunter,
    LoginIncidentContext,
    serialize_login_evidence,
)
from agents.log_hunter import (
    DEFAULT_DATABASE,
    DuckDBLogHunter,
    EvidenceRecord,
    EventSearchFilters,
    IncidentContext,
)
from agents.response_draft import (
    DEFAULT_AUDIT_PATH,
    DryRunDecisionStore,
    draft_response_proposals,
    serialize_proposals,
)


DATABASE_PATH = DEFAULT_DATABASE
LANL_FEATURE_PATH = DEFAULT_FEATURES
AUDIT_PATH = DEFAULT_AUDIT_PATH
MAX_PENDING_PROPOSALS = 1000
PENDING_PROPOSALS: dict[str, dict] = {}

app = FastAPI(title="AI SOC Analyst Copilot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class NetworkSearchRequest(BaseModel):
    incident_id: str = Field(min_length=1, max_length=128)
    entity: str = Field(min_length=1, max_length=255)
    start_time: datetime
    end_time: datetime
    limit: int = Field(default=100, ge=1, le=500)
    src_ip: str | None = None
    dst_ip: str | None = None
    source_port: int | None = Field(default=None, ge=0, le=65535)
    destination_port: int | None = Field(default=None, ge=0, le=65535)
    protocol: str | None = None


class LanlSearchRequest(BaseModel):
    incident_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=255)
    start_hour: int = Field(ge=0)
    end_hour: int = Field(ge=0)
    limit: int = Field(default=100, ge=1, le=500)


class ResponseDecisionRequest(BaseModel):
    proposal_id: str = Field(min_length=1, max_length=64)
    decision: Literal["approve", "reject"]
    analyst: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=2000)


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


def _network_context(request: NetworkSearchRequest) -> IncidentContext:
    return IncidentContext(
        incident_id=request.incident_id,
        entity=request.entity,
        start_time=request.start_time,
        end_time=request.end_time,
    )


def _network_search(request: NetworkSearchRequest) -> dict:
    try:
        incident = _network_context(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    filters = EventSearchFilters(
        src_ip=request.src_ip,
        dst_ip=request.dst_ip,
        source_port=request.source_port,
        destination_port=request.destination_port,
        protocol=request.protocol,
    )
    try:
        evidence = DuckDBLogHunter(DATABASE_PATH).search(
            incident, limit=request.limit, filters=filters
        )
    except (FileNotFoundError, ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    possibly_truncated = len(evidence) == request.limit
    attack_mapping = serialize_mapping_result(
        map_attack_hypotheses(
            incident, evidence, possibly_truncated=possibly_truncated
        )
    )
    query = "network discovery port fan-out"
    if attack_mapping["hypotheses"]:
        query += " T1046 MITRE ATT&CK"
    references = serialize_references(
        LocalKnowledgeBase().search(query, limit=5)
    )
    evidence_payload = [record.__dict__ for record in evidence]
    proposals = serialize_proposals(
        draft_response_proposals(
            incident.incident_id, evidence_payload, attack_mapping
        )
    )
    for proposal in proposals:
        if len(PENDING_PROPOSALS) >= MAX_PENDING_PROPOSALS:
            PENDING_PROPOSALS.pop(next(iter(PENDING_PROPOSALS)))
        PENDING_PROPOSALS[proposal["proposal_id"]] = proposal
    return {
        "incident": {
            "incident_id": incident.incident_id,
            "entity": incident.entity,
            "start_time": incident.start_time.isoformat(sep=" "),
            "end_time": incident.end_time.isoformat(sep=" "),
        },
        "source": "cicids2017",
        "evidence": evidence_payload,
        "evidence_count": len(evidence_payload),
        "possibly_truncated": possibly_truncated,
        "attack_mapping": attack_mapping,
        "knowledge_references": references,
        "response_proposals": proposals,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "execution_mode": "dry_run_only"}


@app.post("/api/search/network")
def search_network(request: NetworkSearchRequest):
    return _network_search(request)


@app.post("/api/search/lanl")
def search_lanl(request: LanlSearchRequest):
    incident = LoginIncidentContext(
        incident_id=request.incident_id,
        user_id=request.user_id,
        start_hour=request.start_hour,
        end_hour=request.end_hour,
    )
    try:
        evidence = LanlFeatureHunter(LANL_FEATURE_PATH).search(
            incident, limit=request.limit
        )
    except (FileNotFoundError, ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "incident": {
            "incident_id": incident.incident_id,
            "user_id": incident.user_id,
            "start_hour": incident.start_hour,
            "end_hour": incident.end_hour,
        },
        "source": "lanl_user_hour_features",
        "time_semantics": "relative hour index, not wall-clock time",
        "evidence": serialize_login_evidence(evidence),
        "evidence_count": len(evidence),
        "possibly_truncated": len(evidence) == request.limit,
        "attack_mapping": {
            "status": "not_evaluated",
            "notice": "Current ATT&CK heuristic applies only to network flows.",
            "hypotheses": [],
        },
        "knowledge_references": [],
        "response_proposals": [],
    }


@app.post("/api/report/network")
def report_network(request: NetworkSearchRequest):
    search = _network_search(request)
    incident = _network_context(request)
    try:
        if not search["evidence"]:
            report = generate_verified_report(
                incident,
                [],
                request.limit,
                client=None,
                knowledge_references=search["knowledge_references"],
            )
        else:
            report = generate_verified_report(
                incident,
                [
                    EvidenceRecord(**record)
                    for record in search["evidence"]
                ],
                request.limit,
                client=GroqChatClient(),
                knowledge_references=search["knowledge_references"],
            )
    except UnsupportedClaimError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    search["report"] = report
    return search


@app.post("/api/report/lanl")
def report_lanl(request: LanlSearchRequest):
    if request.start_hour > request.end_hour:
        raise HTTPException(
            status_code=400, detail="start_hour must not be after end_hour"
        )
    incident = {
        "incident_id": request.incident_id,
        "user_id": request.user_id,
        "start_hour": request.start_hour,
        "end_hour": request.end_hour,
        "time_semantics": "relative hour index, not wall-clock time",
    }
    try:
        evidence = LanlFeatureHunter(LANL_FEATURE_PATH).search(
            LoginIncidentContext(
                incident_id=request.incident_id,
                user_id=request.user_id,
                start_hour=request.start_hour,
                end_hour=request.end_hour,
            ),
            limit=request.limit,
        )
        report = generate_login_report(
            incident,
            evidence,
            request.limit,
            GroqChatClient() if evidence else None,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except UnsupportedClaimError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {
        **incident,
        "source": "lanl_user_hour_features",
        "evidence": report["evidence"],
        "evidence_count": len(evidence),
        "possibly_truncated": len(evidence) == request.limit,
        "attack_mapping": {
            "status": "not_evaluated",
            "notice": "Current ATT&CK heuristic applies only to network flows.",
            "hypotheses": [],
        },
        "knowledge_references": [],
        "response_proposals": [],
        "report": report,
    }


@app.post("/api/knowledge/search")
def search_knowledge(request: KnowledgeSearchRequest):
    return {
        "references": serialize_references(
            LocalKnowledgeBase().search(request.query, request.limit)
        )
    }


@app.post("/api/responses/decision")
def decide_response(request: ResponseDecisionRequest):
    proposal = PENDING_PROPOSALS.get(request.proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="Response proposal not found")
    from agents.response_draft import ResponseProposal

    try:
        decision = DryRunDecisionStore(AUDIT_PATH).record_decision(
            ResponseProposal(**proposal),
            request.decision,
            request.analyst,
            request.reason,
        )
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    PENDING_PROPOSALS.pop(request.proposal_id, None)
    return {
        **decision,
        "notice": "Recorded for dry-run audit only; no response action was executed.",
    }


@app.get("/api/responses/{incident_id}")
def list_response_decisions(incident_id: str):
    return {"decisions": DryRunDecisionStore(AUDIT_PATH).list_decisions(incident_id)}
