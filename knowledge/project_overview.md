# Project Overview: Real Estate Agentic Intelligence System

## Core Objective
This project converts the original ML-only valuation app into an Agentic AI system that combines:
1. A deterministic pricing tool (Random Forest model).
2. Retrieval-Augmented Generation (RAG) over curated project and domain documents.
3. LangGraph orchestration for routing, retrieval, tool calls, grounded reasoning, and guardrails.

## Why Agentic AI Here
Pure regression predicts a number but cannot explain assumptions, model limits, data lineage, or confidence narrative. The Agentic layer solves this by:
- Routing user intent (valuation vs methodology vs market question).
- Pulling relevant evidence from indexed knowledge chunks.
- Calling pricing tools when structured property input exists.
- Generating grounded natural-language outputs with source references.

## Workflow Stages
1. Route user query to a path in the graph.
2. Retrieve top-k context chunks from vector + lexical retrieval.
3. Run pricing tool for quantitative estimate when applicable.
4. Ask Gemini 2.5 Flash to synthesize a grounded response.
5. Apply guardrails to inject uncertainty messaging when retrieval is weak.

## User-Facing Outcomes
- Direct valuation in JPY using trained model.
- Explainable answer with cited retrieval sources.
- Robust fallback behavior when APIs fail.
- Diagnostic tab with benchmark and retrieval transparency.
