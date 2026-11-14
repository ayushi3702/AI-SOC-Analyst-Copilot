import json

import pytest

from agents import knowledge
from knowledge.import_attack import (
    extract_enterprise_techniques,
    import_attack_corpus,
)


def _technique(technique_id, **overrides):
    item = {
        "type": "attack-pattern",
        "name": f"Technique {technique_id}",
        "description": f"Description for {technique_id}.",
        "external_references": [{
            "source_name": "mitre-attack",
            "external_id": technique_id,
            "url": f"https://attack.mitre.org/techniques/{technique_id}/",
        }],
        "kill_chain_phases": [{
            "kill_chain_name": "mitre-attack",
            "phase_name": "discovery",
        }],
        "x_mitre_platforms": ["Windows"],
    }
    return {**item, **overrides}


def _bundle(*objects):
    return {"type": "bundle", "objects": list(objects)}


def test_importer_extracts_active_enterprise_techniques_only():
    records = extract_enterprise_techniques(_bundle(
        _technique("T1046"),
        _technique("T1046.001"),
        _technique("T9999", revoked=True),
        _technique("T8888", x_mitre_deprecated=True),
        {
            **_technique("X1234"),
            "external_references": [{
                "source_name": "other-framework",
                "external_id": "X1234",
            }],
        },
    ))

    assert [record["reference_id"] for record in records] == [
        "mitre:T1046",
        "mitre:T1046.001",
    ]
    assert records[0]["citation"] == "MITRE ATT&CK Enterprise, T1046"
    assert "Tactics: Discovery." in records[0]["text"]
    assert "not evidence" in records[0]["text"]


def test_importer_rejects_invalid_or_empty_bundles():
    with pytest.raises(ValueError, match="STIX bundle"):
        extract_enterprise_techniques({"type": "not-a-bundle", "objects": []})
    with pytest.raises(ValueError, match="No active"):
        extract_enterprise_techniques(_bundle(_technique("T1000", revoked=True)))


def test_importer_writes_corpus_and_local_knowledge_augments_static_refs(tmp_path, monkeypatch):
    bundle_path = tmp_path / "bundle.json"
    corpus_path = tmp_path / "attack.json"
    bundle_path.write_text(
        json.dumps(_bundle(_technique("T1046"), _technique("T9999"))),
        encoding="utf-8",
    )
    assert import_attack_corpus(corpus_path, bundle_path) == 2
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    assert len(corpus) == 2

    static_corpus = tmp_path / "static.json"
    static_corpus.write_text(
        json.dumps([{
            "reference_id": "sigma:test",
            "title": "Sigma test rule",
            "kind": "Sigma-style",
            "citation": "Local test",
            "source_url": "https://example.test/rule",
            "text": "Experimental test rule.",
        }]),
        encoding="utf-8",
    )
    monkeypatch.setattr(knowledge, "DEFAULT_STATIC_CORPUS", static_corpus)
    monkeypatch.setattr(knowledge, "DEFAULT_ATTACK_CORPUS", corpus_path)
    references = knowledge.LocalKnowledgeBase().as_records()
    ids = {reference["reference_id"] for reference in references}
    assert {"mitre:T1046", "mitre:T9999", "sigma:test"} <= ids


def test_importer_validates_timeout(tmp_path):
    with pytest.raises(ValueError, match="timeout"):
        import_attack_corpus(tmp_path / "out.json", timeout=0)
