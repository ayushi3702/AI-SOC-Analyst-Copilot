"""Local TF-IDF retrieval for ATT&CK and Sigma reference documents."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_CORPUS = ROOT / "knowledge" / "references.json"
DEFAULT_ATTACK_CORPUS = ROOT / "data" / "processed" / "mitre_attack_techniques.json"


@dataclass(frozen=True)
class KnowledgeReference:
    reference_id: str
    title: str
    kind: str
    citation: str
    source_url: str
    text: str
    score: float


class LocalKnowledgeBase:
    """Retrieve cited local documents; reference text is not incident evidence."""

    def __init__(self, corpus_path=None):
        if corpus_path is None:
            corpus_paths = [DEFAULT_STATIC_CORPUS]
            if DEFAULT_ATTACK_CORPUS.is_file():
                corpus_paths.append(DEFAULT_ATTACK_CORPUS)
        else:
            corpus_paths = [Path(corpus_path)]
        self.corpus_path = corpus_paths[-1]
        documents_by_id = {}
        for path in corpus_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Knowledge corpus not found: {path}")
            documents = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(documents, list) or not documents:
                raise ValueError("Knowledge corpus must contain a non-empty JSON list")
            source_identifiers = []
            for document in documents:
                if not isinstance(document, dict) or set(document) != {
                    "reference_id",
                    "title",
                    "kind",
                    "citation",
                    "source_url",
                    "text",
                }:
                    raise ValueError("Knowledge document does not match the corpus schema")
                if any(
                    not isinstance(document[key], str) or not document[key].strip()
                    for key in document
                ):
                    raise ValueError("Knowledge document fields must be non-empty strings")
                source_identifiers.append(document["reference_id"])
                documents_by_id[document["reference_id"]] = document
            if len(set(source_identifiers)) != len(source_identifiers):
                raise ValueError("Knowledge reference IDs must be unique")
        documents = list(documents_by_id.values())
        if not isinstance(documents, list) or not documents:
            raise ValueError("Knowledge corpus must contain a non-empty JSON list")
        identifiers = [document["reference_id"] for document in documents]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Knowledge reference IDs must be unique")
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(
            [
                f"{item['title']} {item['kind']} {item['text']}"
                for item in self.documents
            ]
        )

    def search(self, query: str, limit: int = 5) -> list[KnowledgeReference]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("limit must be an integer from 1 to 20")
        vector = self.vectorizer.transform([query])
        scores = cosine_similarity(vector, self.matrix).ravel()
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), self.documents[item[0]]["reference_id"]),
        )
        return [
            KnowledgeReference(**self.documents[index], score=float(score))
            for index, score in ranked[:limit]
            if score > 0
        ]

    def as_records(self) -> list[dict]:
        return [dict(document) for document in self.documents]


def serialize_references(references: list[KnowledgeReference]) -> list[dict]:
    return [asdict(reference) for reference in references]
