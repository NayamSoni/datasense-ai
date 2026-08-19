"""Retrieval helpers for DataSense AI's business knowledge base.

Local development uses Ollama's embedding model for semantic retrieval.
Cloud deployment can use a lightweight TF-IDF index that does not require a
separate embedding service. Documents, indexes, and uploaded knowledge remain
in the current Streamlit session.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
from io import BytesIO, StringIO
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_EMBEDDING_MODEL = os.getenv(
    "DATASENSE_EMBEDDING_MODEL",
    "embeddinggemma",
)
SUPPORTED_KNOWLEDGE_TYPES = ("pdf", "txt", "md", "csv")
SUPPORTED_RETRIEVAL_BACKENDS = ("ollama", "tfidf")

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
STARTER_GLOSSARY_CANDIDATES = (
    DATA_DIRECTORY / "kpi_glossary_10_industries.csv",
    DATA_DIRECTORY / "kpi_glossary_sample.csv",
)
STARTER_GLOSSARY_PATH = next(
    (path for path in STARTER_GLOSSARY_CANDIDATES if path.exists()),
    STARTER_GLOSSARY_CANDIDATES[0],
)


class KnowledgeBaseError(RuntimeError):
    """A user-facing error raised while building or querying knowledge."""


EmbeddingFunction = Callable[[list[str], str], list[list[float]]]


def configured_retrieval_backend() -> str:
    """Select Ollama locally and TF-IDF for direct Ollama Cloud usage."""
    configured_backend = (
        os.getenv("DATASENSE_RAG_BACKEND")
        or os.getenv("RAG_BACKEND")
        or ""
    ).strip().lower()

    if not configured_backend:
        ollama_host = (os.getenv("OLLAMA_HOST") or "").rstrip("/")
        configured_backend = (
            "tfidf"
            if ollama_host == "https://ollama.com"
            else "ollama"
        )

    if configured_backend not in SUPPORTED_RETRIEVAL_BACKENDS:
        supported = ", ".join(SUPPORTED_RETRIEVAL_BACKENDS)
        raise KnowledgeBaseError(
            f"Unsupported RAG backend '{configured_backend}'. "
            f"Supported backends: {supported}."
        )

    return configured_backend


def _normalise_text(text: str) -> str:
    text = str(text).replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise KnowledgeBaseError("The text file encoding could not be read.")


def _starter_glossary_rows(
    path: Path = STARTER_GLOSSARY_PATH,
) -> tuple[list[str], list[dict]]:
    """Read the bundled multi-industry glossary with contract checks."""
    try:
        text = _decode_text(path.read_bytes())
    except OSError as exc:
        raise KnowledgeBaseError(
            f"Could not read the starter glossary: {exc}"
        ) from exc

    reader = csv.DictReader(StringIO(text))
    fieldnames = list(reader.fieldnames or [])
    required = {"industry", "term", "definition", "formula"}
    if not required.issubset(fieldnames):
        missing = ", ".join(sorted(required.difference(fieldnames)))
        raise KnowledgeBaseError(
            f"Starter glossary is missing columns: {missing}"
        )

    rows = [
        row
        for row in reader
        if any(str(value).strip() for value in row.values())
    ]
    return fieldnames, rows


def available_starter_industries(
    path: Path = STARTER_GLOSSARY_PATH,
) -> list[str]:
    """Return bundled industries in their curated display order."""
    _, rows = _starter_glossary_rows(path)
    return list(
        dict.fromkeys(row["industry"].strip() for row in rows)
    )


def starter_glossary_document(
    industry: str,
    path: Path = STARTER_GLOSSARY_PATH,
) -> tuple[str, bytes]:
    """Create an upload-shaped CSV containing one starter industry."""
    fieldnames, rows = _starter_glossary_rows(path)
    selected_rows = [
        row
        for row in rows
        if row["industry"].strip().casefold()
        == industry.strip().casefold()
    ]
    if not selected_rows:
        raise KnowledgeBaseError(
            f"No starter knowledge exists for '{industry}'."
        )

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(selected_rows)

    safe_industry = re.sub(
        r"[^a-z0-9]+",
        "_",
        industry.lower(),
    ).strip("_")
    name = f"datasense_{safe_industry}_starter_kpis.csv"
    return name, output.getvalue().encode("utf-8")


def _csv_sections(name: str, data: bytes) -> list[dict[str, str]]:
    text = _decode_text(data)
    try:
        reader = csv.DictReader(StringIO(text))
        fields = [
            (field, str(field).strip())
            for field in (reader.fieldnames or [])
        ]
    except csv.Error as exc:
        raise KnowledgeBaseError(
            f"Could not read {name} as CSV: {exc}"
        ) from exc

    if not fields:
        raise KnowledgeBaseError(
            f"{name} does not contain a CSV header row."
        )

    sections = []
    for row_number, row in enumerate(reader, start=2):
        values = []
        for raw_field, label in fields:
            value = _normalise_text(row.get(raw_field, ""))
            if value:
                values.append(f"{label}: {value}")
        if values:
            sections.append({
                "source": name,
                "location": f"row {row_number}",
                "text": "\n".join(values),
            })
    return sections


def _pdf_sections(name: str, data: bytes) -> list[dict[str, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise KnowledgeBaseError(
            "PDF support needs pypdf. "
            "Run: pip install -r requirements.txt"
        ) from exc

    try:
        reader = PdfReader(BytesIO(data))
    except Exception as exc:
        raise KnowledgeBaseError(
            f"Could not read {name} as PDF: {exc}"
        ) from exc

    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _normalise_text(page.extract_text() or "")
        if text:
            sections.append({
                "source": name,
                "location": f"page {page_number}",
                "text": text,
            })
    return sections


def extract_document(name: str, data: bytes) -> list[dict[str, str]]:
    """Extract source-labelled text sections from a supported document."""
    extension = Path(name).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_KNOWLEDGE_TYPES:
        supported = ", ".join(SUPPORTED_KNOWLEDGE_TYPES)
        raise KnowledgeBaseError(
            f"{name} is unsupported. Use: {supported}."
        )

    if extension == "csv":
        return _csv_sections(name, data)
    if extension == "pdf":
        return _pdf_sections(name, data)

    text = _normalise_text(_decode_text(data))
    if not text:
        return []
    return [{
        "source": name,
        "location": "document",
        "text": text,
    }]


def _chunk_section(
    section: dict[str, str],
    chunk_words: int,
    overlap_words: int,
) -> list[dict[str, str]]:
    words = section["text"].split()
    if not words:
        return []

    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        part = words[start:start + chunk_words]
        if not part:
            break
        text = " ".join(part)
        identity = (
            f"{section['source']}|{section['location']}|{start}|{text}"
        )
        chunks.append({
            "id": hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:16],
            "source": section["source"],
            "location": section["location"],
            "text": text,
        })
        if start + chunk_words >= len(words):
            break
    return chunks


def chunk_documents(
    documents: Iterable[tuple[str, bytes]],
    chunk_words: int = 180,
    overlap_words: int = 30,
) -> list[dict[str, str]]:
    """Extract and chunk documents while preserving source labels."""
    if chunk_words < 20:
        raise ValueError("chunk_words must be at least 20")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError(
            "overlap_words must be between 0 and chunk_words - 1"
        )

    chunks = []
    for name, data in documents:
        for section in extract_document(name, data):
            chunks.extend(
                _chunk_section(section, chunk_words, overlap_words)
            )
    return chunks


def ollama_embed_texts(
    texts: list[str],
    model: str,
) -> list[list[float]]:
    """Generate local semantic embeddings using Ollama."""
    try:
        import ollama

        response = ollama.embed(model=model, input=texts)
    except Exception as exc:
        raise KnowledgeBaseError(
            f"Could not use the local embedding model '{model}'. "
            f"Make sure Ollama is running, then run: "
            f"ollama pull {model}. Details: {exc}"
        ) from exc

    embeddings = (
        response.get("embeddings")
        if isinstance(response, dict)
        else getattr(response, "embeddings", None)
    )
    if not embeddings:
        raise KnowledgeBaseError(
            "Ollama returned no document embeddings."
        )
    return [list(map(float, vector)) for vector in embeddings]


def _build_tfidf_index(
    chunks: list[dict[str, str]],
    document_names: list[str],
) -> dict:
    """Build a lightweight lexical retrieval index for cloud usage."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise KnowledgeBaseError(
            "TF-IDF retrieval requires scikit-learn. "
            "Run: pip install -r requirements.txt"
        ) from exc

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=20_000,
    )

    try:
        matrix = vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )
    except ValueError as exc:
        raise KnowledgeBaseError(
            "The knowledge documents did not contain enough searchable text."
        ) from exc

    return {
        "backend": "tfidf",
        "model": "tfidf-word-bigram",
        "chunks": chunks,
        "tfidf_vectorizer": vectorizer,
        "tfidf_matrix": matrix,
        "documents": document_names,
    }


def _unit_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise KnowledgeBaseError(
            "The embedding model returned an empty vector."
        )
    return [value / magnitude for value in vector]


def build_knowledge_index(
    documents: Iterable[tuple[str, bytes]],
    model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_function: EmbeddingFunction = ollama_embed_texts,
    retrieval_backend: str | None = None,
) -> dict:
    """Build a semantic or TF-IDF index for uploaded knowledge."""
    document_list = list(documents)
    chunks = chunk_documents(document_list)
    if not chunks:
        raise KnowledgeBaseError(
            "No readable text was found in the uploaded files."
        )

    backend = (
        retrieval_backend or configured_retrieval_backend()
    ).strip().lower()
    if backend not in SUPPORTED_RETRIEVAL_BACKENDS:
        supported = ", ".join(SUPPORTED_RETRIEVAL_BACKENDS)
        raise KnowledgeBaseError(
            f"Unsupported RAG backend '{backend}'. "
            f"Supported backends: {supported}."
        )

    document_names = [name for name, _ in document_list]
    if backend == "tfidf":
        return _build_tfidf_index(chunks, document_names)

    embeddings = embedding_function(
        [chunk["text"] for chunk in chunks],
        model,
    )
    if len(embeddings) != len(chunks):
        raise KnowledgeBaseError(
            "The embedding count did not match the document chunks."
        )

    normalised = [_unit_vector(vector) for vector in embeddings]
    dimensions = len(normalised[0])
    if any(len(vector) != dimensions for vector in normalised):
        raise KnowledgeBaseError(
            "The embedding model returned inconsistent dimensions."
        )

    return {
        "backend": "ollama",
        "model": model,
        "chunks": chunks,
        "embeddings": normalised,
        "documents": document_names,
    }


def _retrieve_with_tfidf(
    question: str,
    index: dict,
    top_k: int,
    min_score: float,
) -> list[dict]:
    """Retrieve chunks using cosine similarity over TF-IDF vectors."""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise KnowledgeBaseError(
            "TF-IDF retrieval requires scikit-learn. "
            "Run: pip install -r requirements.txt"
        ) from exc

    vectorizer = index.get("tfidf_vectorizer")
    matrix = index.get("tfidf_matrix")
    if vectorizer is None or matrix is None:
        raise KnowledgeBaseError(
            "The TF-IDF knowledge index is incomplete. Rebuild the index."
        )

    query = vectorizer.transform([question])
    if query.nnz == 0:
        return []

    scores = cosine_similarity(query, matrix).ravel()
    scored = []
    for chunk, score in zip(index["chunks"], scores):
        numeric_score = float(score)
        if numeric_score >= min_score:
            scored.append({**chunk, "score": numeric_score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:max(1, top_k)]


def retrieve_knowledge(
    question: str,
    index: dict,
    top_k: int = 3,
    min_score: float | None = None,
    embedding_function: EmbeddingFunction = ollama_embed_texts,
) -> list[dict]:
    """Return the most relevant source chunks for a question."""
    if not question.strip() or not index.get("chunks"):
        return []

    backend = str(index.get("backend", "ollama")).strip().lower()
    if backend == "tfidf":
        threshold = 0.05 if min_score is None else min_score
        return _retrieve_with_tfidf(
            question,
            index,
            top_k,
            threshold,
        )

    if backend != "ollama":
        raise KnowledgeBaseError(
            f"The knowledge index uses an unsupported backend: {backend}."
        )

    threshold = 0.15 if min_score is None else min_score
    query_vectors = embedding_function([question], index["model"])
    if not query_vectors:
        raise KnowledgeBaseError(
            "Ollama returned no query embedding."
        )
    query = _unit_vector(query_vectors[0])

    embeddings = index.get("embeddings", [])
    if embeddings and len(query) != len(embeddings[0]):
        raise KnowledgeBaseError(
            "The query embedding does not match the knowledge index. "
            "Rebuild the index."
        )

    scored = []
    for chunk, vector in zip(index["chunks"], embeddings):
        score = sum(
            left * right
            for left, right in zip(query, vector)
        )
        if score >= threshold:
            scored.append({**chunk, "score": float(score)})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:max(1, top_k)]


def format_retrieved_context(results: list[dict]) -> str:
    """Create numbered evidence blocks that the answer can cite."""
    blocks = []
    for number, result in enumerate(results, start=1):
        label = f"{result['source']}, {result['location']}"
        blocks.append(
            f"[Source {number} — {label}]\n{result['text']}"
        )
    return "\n\n".join(blocks)
