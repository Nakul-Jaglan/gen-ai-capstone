# RAG Design Notes

## Corpus
The retrieval corpus uses repository-local knowledge documents under `knowledge/`.
This keeps answers aligned with actual project constraints and prevents open-web drift.

## Chunking
Documents are chunked into overlapping windows so each chunk preserves enough local context while remaining embedding-friendly.

## Retrieval Strategy
Hybrid retrieval score:
- Vector similarity from Gemini embeddings (`gemini-embedding-001`).
- Lexical token overlap score for robustness.
- Weighted fusion:

combined_score = alpha * vector_score + (1 - alpha) * lexical_score

where `alpha = 0.75` by default.

## Why Hybrid
Vector retrieval handles semantic paraphrases, while lexical overlap catches exact term matches (feature names, metrics, model IDs). Hybrid retrieval improves precision for technical QA.

## Caching and Performance
- The index is persisted to `.cache/rag_index.joblib`.
- Rebuild occurs only when corpus file signatures change.
- This reduces startup latency and repeated embedding API calls.

## Failure Handling
If embedding API fails:
- System falls back to lexical retrieval.
- Agent response includes explicit low-confidence warning.
- User still receives deterministic information where possible.
