"""Centralised configuration for the Real Estate Concierge application."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


def load_env_file(env_path: Path) -> None:
    """Load key-value pairs from a .env file into process environment."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    """Immutable settings object populated from environment variables and .env file."""
    project_root: Path
    data_path: Path
    model_path: Path
    scaler_path: Path
    knowledge_dir: Path
    cache_dir: Path
    index_path: Path
    eval_questions_path: Path
    env_path: Path

    groq_api_key: str | None
    generation_model: str
    embedding_model: str

    top_k: int
    retrieval_alpha: float
    max_context_chars: int
    temperature: float
    max_output_tokens: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Build and cache a Settings instance.
    Loads variables from the project-root .env file before reading environment.
    """
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    load_env_file(env_path)

    cache_dir = project_root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    groq_api_key = os.getenv("GROQ_API_KEY")

    return Settings(
        project_root=project_root,
        data_path=project_root / "02.csv",
        model_path=project_root / "rf_model_new.joblib",
        scaler_path=project_root / "minmaxscaler.joblib",
        knowledge_dir=project_root / "knowledge",
        cache_dir=cache_dir,
        index_path=cache_dir / "rag_index.joblib",
        eval_questions_path=project_root / "data" / "eval_questions.json",
        env_path=env_path,
        groq_api_key=groq_api_key,
        generation_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        embedding_model=os.getenv("RAG_EMBED_MODEL", "lexical-only"),
        top_k=int(os.getenv("RAG_TOP_K", "4")),
        retrieval_alpha=float(os.getenv("RAG_ALPHA", "0.75")),
        max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "5000")),
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")),
        max_output_tokens=int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "600")),
    )
