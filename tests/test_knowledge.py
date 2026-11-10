import json

import pytest

from agents.knowledge import LocalKnowledgeBase


def test_local_knowledge_search_returns_cited_reference_records():
    references = LocalKnowledgeBase().search("MITRE network service discovery", limit=3)

    assert references
    assert references[0].reference_id == "mitre:T1046"
    assert references[0].source_url == "https://attack.mitre.org/techniques/T1046/"
    assert references[0].score > 0


def test_local_knowledge_supports_sigma_retrieval():
    references = LocalKnowledgeBase().search("Sigma experimental network port fan-out")

    assert any(reference.reference_id == "sigma:local-port-fanout"
               for reference in references)


def test_local_knowledge_validates_query_and_corpus_schema(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        LocalKnowledgeBase().search(" ")

    corpus = tmp_path / "references.json"
    corpus.write_text(json.dumps([{"reference_id": "bad"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        LocalKnowledgeBase(corpus)
