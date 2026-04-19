from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.llm import LLMClient, LLMError
from src.pricing import PricingEngine
from src.rag import RAGEngine, RetrievalHit


class AgentState(TypedDict, total=False):
    question: str
    property_payload: dict[str, Any] | None

    route: str
    retrieval_hits: list[RetrievalHit]
    prediction_jpy: float | None
    prediction_warnings: list[str]
    inferred_fields: list[str]

    analysis_answer: str
    analysis_source: str

    draft_answer: str
    final_answer: str
    warnings: list[str]
    confidence: float
    citations: list[str]


@dataclass
class AgentResult:
    answer: str
    route: str
    confidence: float
    citations: list[str]
    warnings: list[str]
    retrieval_hits: list[RetrievalHit]
    prediction_jpy: float | None
    latency_ms: float


class RealEstateAgent:
    """LangGraph-based orchestrator for route -> retrieve -> tools -> reason -> guardrail."""

    PRICING_KEYWORDS = {
        "price",
        "predict",
        "valuation",
        "estimate",
        "worth",
        "cost",
    }

    PRICING_PHRASES = {"how much", "expected price", "property value"}

    ANALYTICS_METRIC_KEYWORDS = {
        "highest",
        "lowest",
        "average",
        "mean",
        "median",
        "trend",
        "top",
        "cheapest",
        "most expensive",
        "overall average",
        "by year",
        "over year",
    }

    ANALYTICS_DIMENSION_KEYWORDS = {
        "region",
        "area",
        "municipality",
        "district",
        "station",
    }

    INVESTMENT_KEYWORDS = {
        "invest",
        "investment",
        "profit",
        "return",
        "roi",
        "appreciation",
        "growth potential",
        "undervalued",
        "best place to buy",
        "opportunity",
    }

    GREETING_KEYWORDS = {"hi", "hello", "hey"}
    GREETING_PHRASES = {"good morning", "good evening", "good afternoon"}

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        rag: RAGEngine,
        pricing: PricingEngine,
    ):
        self.settings = settings
        self.llm = llm
        self.rag = rag
        self.pricing = pricing

        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("route", self._route_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("analysis", self._analysis_node)
        builder.add_node("pricing", self._pricing_node)
        builder.add_node("reason", self._reason_node)
        builder.add_node("guardrail", self._guardrail_node)

        builder.add_edge(START, "route")
        builder.add_conditional_edges(
            "route",
            self._route_decision,
            {
                "greeting": "reason",
                "knowledge": "retrieve",
                "analytics": "analysis",
                "pricing": "retrieve",
            },
        )

        builder.add_conditional_edges(
            "retrieve",
            self._post_retrieve_decision,
            {
                "pricing": "pricing",
                "knowledge": "reason",
            },
        )

        builder.add_edge("analysis", "reason")
        builder.add_edge("pricing", "reason")
        builder.add_edge("reason", "guardrail")
        builder.add_edge("guardrail", END)

        return builder.compile()

    def _route_node(self, state: AgentState) -> AgentState:
        question = (state.get("question") or "").strip().lower()
        has_payload = bool(state.get("property_payload"))
        tokens = set(re.findall(r"[a-z]+", question))

        is_greeting = bool(tokens & self.GREETING_KEYWORDS) or any(
            phrase in question for phrase in self.GREETING_PHRASES
        )
        pricing_intent = bool(tokens & self.PRICING_KEYWORDS) or any(
            phrase in question for phrase in self.PRICING_PHRASES
        )

        inferred_payload = None
        if pricing_intent and not has_payload:
            inferred_payload, _ = self.pricing.payload_from_text(question)
        has_inferred_payload = inferred_payload is not None

        has_metric = any(keyword in question for keyword in self.ANALYTICS_METRIC_KEYWORDS)
        has_dimension = any(keyword in question for keyword in self.ANALYTICS_DIMENSION_KEYWORDS)
        investment_intent = any(keyword in question for keyword in self.INVESTMENT_KEYWORDS)
        analytics_intent = investment_intent or (has_metric and (has_dimension or "price" in question))

        if is_greeting:
            route = "greeting"
        elif pricing_intent and (has_payload or has_inferred_payload):
            route = "pricing"
        elif analytics_intent:
            route = "analytics"
        elif pricing_intent:
            route = "pricing"
        else:
            route = "knowledge"

        return {
            "route": route,
            "prediction_jpy": None,
            "prediction_warnings": [],
            "inferred_fields": [],
            "analysis_answer": "",
            "analysis_source": "",
            "warnings": [],
            "retrieval_hits": [],
            "citations": [],
            "confidence": 0.0,
        }

    def _route_decision(self, state: AgentState) -> str:
        return state.get("route", "knowledge")

    def _post_retrieve_decision(self, state: AgentState) -> str:
        return "pricing" if state.get("route") == "pricing" else "knowledge"

    def _retrieve_node(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        hits = self.rag.search(question, top_k=self.settings.top_k)
        return {"retrieval_hits": hits}

    def _analysis_node(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        result = self.pricing.answer_market_query(question)
        if not result.handled:
            return {}
        return {
            "analysis_answer": result.answer,
            "analysis_source": result.source,
        }

    def _pricing_node(self, state: AgentState) -> AgentState:
        payload = state.get("property_payload")
        inferred_fields: list[str] = []

        if not payload:
            payload, inferred_fields = self.pricing.payload_from_text(state.get("question", ""))

        if not payload:
            return {
                "prediction_jpy": None,
                "inferred_fields": [],
                "prediction_warnings": [
                    "I need property details to estimate a specific price (for example: land area, floor area, building year, station time, and region/municipality)."
                ],
            }

        result = self.pricing.predict(payload)
        warnings = list(result.warnings)
        if inferred_fields:
            warnings.insert(0, f"Inferred property details from your message: {', '.join(inferred_fields)}.")

        return {
            "prediction_jpy": result.price_jpy,
            "inferred_fields": inferred_fields,
            "prediction_warnings": warnings,
        }

    def _deterministic_fallback(self, state: AgentState) -> str:
        hits = state.get("retrieval_hits", [])
        prediction = state.get("prediction_jpy")
        question = state.get("question", "")

        lines = [
            "I am using a fallback response because Gemini generation is unavailable right now.",
            f"Your question: {question}",
        ]

        if prediction is not None:
            lines.append(f"Estimated property price: JPY {prediction:,.0f}")

        if hits:
            lines.append("Most relevant retrieved evidence:")
            for idx, hit in enumerate(hits[:3], start=1):
                snippet = hit.text[:260].strip()
                lines.append(f"[{idx}] ({hit.source}) {snippet}")
        else:
            lines.append("No strong retrieval evidence was found for this query.")

        lines.append(
            "Please retry after API availability is restored for a fuller natural-language answer."
        )
        return "\n".join(lines)

    def _reason_node(self, state: AgentState) -> AgentState:
        route = state.get("route", "knowledge")
        question = state.get("question", "")
        hits = state.get("retrieval_hits", [])
        prediction = state.get("prediction_jpy")
        prediction_warnings = state.get("prediction_warnings", [])
        analysis_answer = state.get("analysis_answer", "")
        warnings = list(state.get("warnings", []))

        if route == "greeting":
            return {
                "draft_answer": (
                    "Hi. I can help you with property price estimates and market insights from transaction data. "
                    "Ask naturally, for example: 'Which region has the highest average price?'"
                )
            }

        if route == "analytics" and analysis_answer:
            return {"draft_answer": analysis_answer}

        if route == "pricing" and prediction is None:
            guidance = (
                "I can estimate a property price, but I need a few details in your message. "
                "Please include land area, floor area, building year, station walking time, and location (region/municipality)."
            )
            return {"draft_answer": guidance, "warnings": warnings + prediction_warnings}

        context_block = self.rag.build_context_block(hits)

        prompt = (
            "User question:\n"
            f"{question}\n\n"
            "Retrieved context:\n"
            f"{context_block if context_block else 'No retrieved context available.'}\n\n"
            "Pricing tool output:\n"
            f"{f'Predicted price = JPY {prediction:,.0f}' if prediction is not None else 'No pricing output.'}\n"
            f"Pricing warnings = {prediction_warnings if prediction_warnings else 'None'}\n\n"
            "Analytics tool output:\n"
            f"{analysis_answer if analysis_answer else 'No analytics output.'}\n\n"
            "Instructions:\n"
            "1) Answer in concise professional style.\n"
            "2) Prioritize tool outputs when they are available.\n"
            "3) If evidence is weak or missing, explicitly say uncertainty and avoid unsupported claims.\n"
            "4) If pricing output exists, include it in the final answer.\n"
        )

        system_prompt = (
            "You are an expert real-estate analytics copilot. "
            "Stay grounded in provided retrieval context and tool outputs. "
            "Never fabricate sources or metrics."
        )

        if not self.llm.configured:
            warnings.append("Groq API key missing. Returned deterministic fallback answer.")
            return {"draft_answer": self._deterministic_fallback(state), "warnings": warnings}

        try:
            answer = self.llm.generate(prompt, system_prompt=system_prompt).text
            return {"draft_answer": answer, "warnings": warnings}
        except LLMError as exc:
            warnings.append(str(exc))
            return {"draft_answer": self._deterministic_fallback(state), "warnings": warnings}

    def _guardrail_node(self, state: AgentState) -> AgentState:
        hits = state.get("retrieval_hits", [])
        draft_answer = (state.get("draft_answer") or "").strip()
        prediction = state.get("prediction_jpy")
        analysis_source = state.get("analysis_source", "")
        analysis_answer = state.get("analysis_answer", "")
        warnings = list(state.get("warnings", []))

        citations = [f"{hit.source}#chunk-{hit.chunk_id}" for hit in hits]
        if analysis_source:
            citations.append(analysis_source)

        if not hits and prediction is None and not analysis_answer:
            warnings.append(
                "Low confidence: no retrieval evidence and no pricing tool output were available."
            )

        top_score = max((hit.score for hit in hits), default=0.0)
        confidence = 0.30 + min(max(top_score, 0.0), 0.60)
        if prediction is not None:
            confidence += 0.10
        if analysis_answer:
            confidence = max(confidence, 0.85)
        confidence = max(0.05, min(0.98, confidence))

        final_answer = re.sub(r"\[S\d+(?:,\s*S\d+)*\]", "", draft_answer)
        final_answer = re.sub(r"\[Pricing tool output\]", "", final_answer, flags=re.IGNORECASE)
        final_answer = re.sub(r" {2,}", " ", final_answer)
        final_answer = re.sub(r"\s+([.,;:])", r"\1", final_answer).strip()

        return {
            "final_answer": final_answer,
            "citations": citations,
            "warnings": warnings,
            "confidence": float(confidence),
        }

    def ask(self, question: str, property_payload: dict[str, Any] | None = None) -> AgentResult:
        start = time.perf_counter()
        state: AgentState = {
            "question": question,
            "property_payload": property_payload,
        }
        output = self.graph.invoke(state)
        latency_ms = (time.perf_counter() - start) * 1000

        return AgentResult(
            answer=output.get("final_answer", "No answer generated."),
            route=output.get("route", "knowledge"),
            confidence=float(output.get("confidence", 0.0)),
            citations=output.get("citations", []),
            warnings=output.get("warnings", []),
            retrieval_hits=output.get("retrieval_hits", []),
            prediction_jpy=output.get("prediction_jpy"),
            latency_ms=latency_ms,
        )
