"""Hybrid RAG engine combining optional vector-embedding retrieval with lexical (BM25-like) scoring."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import re
from typing import Iterable

import joblib
import numpy as np

from src.config import Settings
from src.llm import LLMClient, LLMError

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class Chunk:
    chunk_id: int
    source: str
    text: str


@dataclass
class RetrievalHit:
    chunk_id: int
    source: str
    text: str
    score: float
    vector_score: float
    lexical_score: float


class RAGEngine:
    """Hybrid retriever with optional vector embeddings + lexical fallback."""

    def __init__(self, settings: Settings, llm: LLMClient):
        self.settings = settings
        self.llm = llm
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.normalized_embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.embedding_available = False

    def _knowledge_files(self) -> list[Path]:
        """Return sorted Markdown knowledge files, falling back to README.md if none are found."""
        files = sorted(self.settings.knowledge_dir.glob("*.md"))
        if files:
            return files

        fallback = []
        readme = self.settings.project_root / "README.md"
        if readme.exists():
            fallback.append(readme)
        return fallback

    @staticmethod
    def _file_signature(files: Iterable[Path]) -> str:
        """Compute a SHA-256 fingerprint of file names, sizes, and modification times."""
        payload = []
        for path in files:
            stat = path.stat()
            payload.append(
                {
                    "name": path.name,
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                }
            )
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _chunk_text(text: str, chunk_words: int = 180, overlap_words: int = 40) -> list[str]:
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_words)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(words):
                break
            start = max(start + chunk_words - overlap_words, start + 1)
        return chunks

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t.lower() for t in _TOKEN_RE.findall(text.lower())}

    @classmethod
    def _lexical_score(cls, query: str, doc: str) -> float:
        q_tokens = cls._tokenize(query)
        d_tokens = cls._tokenize(doc)
        if not q_tokens or not d_tokens:
            return 0.0

        intersection = len(q_tokens & d_tokens)
        if intersection == 0:
            return 0.0

        # Cosine-like token overlap score.
        return intersection / math.sqrt(len(q_tokens) * len(d_tokens))

    def _normalize_embeddings(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms

    def _compute_vector_scores(self, query: str) -> np.ndarray:
        """Return cosine-similarity scores between the query and all indexed chunks.
        Falls back to an all-zeros array when embeddings are unavailable."""
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        if not (self.embedding_available and self.normalized_embeddings.size and self.llm.configured):
            return scores
        try:
            q_vec = self.llm.embed_texts([query])
            if q_vec.size:
                q_norm = q_vec[0] / (np.linalg.norm(q_vec[0]) + 1e-12)
                scores = self.normalized_embeddings @ q_norm
        except LLMError:
            pass
        return scores

    def _build_chunks(self) -> list[Chunk]:
        """Read knowledge files and split them into overlapping word-level chunks."""
        files = self._knowledge_files()
        all_chunks: list[Chunk] = []
        chunk_id = 0

        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            pieces = self._chunk_text(text)
            for piece in pieces:
                all_chunks.append(Chunk(chunk_id=chunk_id, source=path.name, text=piece))
                chunk_id += 1

        return all_chunks

    def ensure_index(self) -> None:
        files = self._knowledge_files()
        signature = self._file_signature(files)

        if self.settings.index_path.exists():
            cached = joblib.load(self.settings.index_path)
            if (
                cached.get("signature") == signature
                and cached.get("embedding_model") == self.settings.embedding_model
            ):
                self.chunks = [Chunk(**item) for item in cached.get("chunks", [])]
                self.embeddings = cached.get("embeddings", np.empty((0, 0), dtype=np.float32))
                self.embedding_available = bool(cached.get("embedding_available", False))
                if self.embedding_available and self.embeddings.size:
                    self.normalized_embeddings = self._normalize_embeddings(self.embeddings)
                else:
                    self.normalized_embeddings = np.empty((0, 0), dtype=np.float32)
                return

        self.chunks = self._build_chunks()
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.normalized_embeddings = np.empty((0, 0), dtype=np.float32)
        self.embedding_available = False

        if self.chunks and self.llm.configured:
            try:
                vectors = self.llm.embed_texts(chunk.text for chunk in self.chunks)
                if len(vectors) == len(self.chunks):
                    self.embeddings = vectors
                    self.normalized_embeddings = self._normalize_embeddings(vectors)
                    self.embedding_available = True
            except LLMError:
                self.embedding_available = False

        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "signature": signature,
                "embedding_model": self.settings.embedding_model,
                "embedding_available": self.embedding_available,
                "chunks": [asdict(chunk) for chunk in self.chunks],
                "embeddings": self.embeddings,
            },
            self.settings.index_path,
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        """Retrieve the top-k most relevant chunks for a query using hybrid scoring."""
        if not self.chunks:
            self.ensure_index()
        if not self.chunks:
            return []

        k = top_k or self.settings.top_k
        lexical_scores = np.array([self._lexical_score(query, c.text) for c in self.chunks], dtype=np.float32)
        vector_scores = self._compute_vector_scores(query)

        alpha = float(self.settings.retrieval_alpha)
        if self.embedding_available and np.any(vector_scores):
            combined = alpha * vector_scores + (1.0 - alpha) * lexical_scores
        else:
            combined = lexical_scores

        ranked_idx = np.argsort(combined)[::-1]
        hits: list[RetrievalHit] = []
        for idx in ranked_idx[: max(k * 2, k)]:
            score = float(combined[idx])
            if score <= 0 and len(hits) >= k:
                continue
            chunk = self.chunks[int(idx)]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=score,
                    vector_score=float(vector_scores[idx]),
                    lexical_score=float(lexical_scores[idx]),
                )
            )
            if len(hits) >= k:
                break

        if not hits:
            top_idx = int(ranked_idx[0])
            chunk = self.chunks[top_idx]
            hits = [
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=float(combined[top_idx]),
                    vector_score=float(vector_scores[top_idx]),
                    lexical_score=float(lexical_scores[top_idx]),
                )
            ]

        return hits

    def build_context_block(self, hits: list[RetrievalHit]) -> str:
        """Format retrieval hits into a numbered context block for the LLM prompt."""
        if not hits:
            return ""

        blocks: list[str] = []
        char_count = 0
        for idx, hit in enumerate(hits, start=1):
            snippet = hit.text.strip()
            chunk_header = (
                f"[S{idx}] source={hit.source} chunk={hit.chunk_id} "
                f"score={hit.score:.3f}\n{snippet}\n"
            )
            if char_count + len(chunk_header) > self.settings.max_context_chars:
                break
            blocks.append(chunk_header)
            char_count += len(chunk_header)

        return "\n".join(blocks)
