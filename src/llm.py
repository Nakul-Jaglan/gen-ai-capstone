from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from groq import Groq

from src.config import Settings


class LLMError(RuntimeError):
    """Raised for recoverable LLM API failures."""


@dataclass
class GenerateResult:
    text: str
    model: str


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Groq | None = None

        if settings.groq_api_key:
            self._client = Groq(api_key=settings.groq_api_key)

    @property
    def configured(self) -> bool:
        return self._client is not None

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> GenerateResult:
        if not self._client:
            raise LLMError("GROQ_API_KEY is missing. Add it to .env.")

        effective_temperature = temperature if temperature is not None else self.settings.temperature
        effective_max_tokens = (
            max_output_tokens if max_output_tokens is not None else self.settings.max_output_tokens
        )

        try:
            response = self._client.chat.completions.create(
                model=self.settings.generation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
        except Exception as exc:  # pragma: no cover - external API
            raise LLMError(f"Groq generation failed: {exc}") from exc

        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not text:
            raise LLMError("Groq returned an empty response.")

        return GenerateResult(text=text, model=self.settings.generation_model)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        if not self._client:
            raise LLMError("GROQ_API_KEY is missing. Add it to .env.")

        items = [t.strip() for t in texts if t and t.strip()]
        if not items:
            return np.empty((0, 0), dtype=np.float32)

        raise LLMError(
            "Embeddings are disabled for the Groq client in this build; using lexical retrieval fallback."
        )


class GeminiClient(LLMClient):
    """Backward-compatible alias for older imports."""
