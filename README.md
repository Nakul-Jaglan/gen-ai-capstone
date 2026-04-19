# Real Estate Agentic Intelligence System (End-Semester Project)

A production-style **Agentic AI + RAG** application for Japanese residential property analytics.

This project upgrades the earlier ML-only app into a full **GenAI system** that combines:
- Deterministic property valuation (Random Forest tool)
- Hybrid retrieval (Gemini embeddings + lexical retrieval)
- LangGraph orchestration for multi-step reasoning
- Gemini 2.5 Flash grounded response generation
- Evaluation dashboard for retrieval/reasoning quality

---

## 1. Technical Implementation (Rubric Mapping)

### 1.1 Correctness and Completeness
The system solves two core problem classes:
1. **Quantitative valuation** using the trained RF model (`rf_model_new.joblib`)
2. **Qualitative reasoning/explanation** with retrieval-grounded LLM answers

### 1.2 Technical Depth
Implemented advanced GenAI topics:
- **RAG**: indexed local project knowledge chunks with source attribution
- **LangGraph**: explicit graph workflow for route → retrieve → tool → reason → guardrail
- **Agentic tools**: pricing tool + retrieval tool orchestrated per query intent

### 1.3 Design Choices
- **LLM**: `gemini-2.5-flash` for low latency + high throughput
- **Embedding model**: `gemini-embedding-001`
- **Vector DB strategy**: local cached embedding index (`.cache/rag_index.joblib`) + hybrid lexical fallback
- **Prompting strategy**: system prompt enforces grounding, uncertainty disclosure, and citation behavior

### 1.4 Performance and Robustness
- Cached retrieval index for faster startup after first build
- Lexical fallback when embedding API is unavailable
- Deterministic fallback answer when generation API fails
- Guardrail node injects low-confidence warning on weak retrieval

---

## 2. Repository Structure and Code Quality

```text
genai/
├── app.py                      # Streamlit frontend (chat + tools + diagnostics)
├── src/
│   ├── config.py               # settings + .env loading
│   ├── llm.py                  # Gemini generation + embedding client
│   ├── rag.py                  # chunking, hybrid retrieval, index cache
│   ├── pricing.py              # RF inference pipeline as deterministic tool
│   ├── agent.py                # LangGraph workflow and guardrails
│   └── evaluation.py           # benchmark evaluation utilities
├── knowledge/
│   ├── project_overview.md     # project/system knowledge for RAG
│   ├── rag_design.md           # retrieval design rationale
│   └── model_and_data_card.md  # model/data limitations and usage notes
├── data/
│   └── eval_questions.json     # benchmark question set
├── 02.csv
├── rf_model_new.joblib
├── minmaxscaler.joblib
├── report.tex
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 3. Setup and Run

## 3.1 Prerequisites
- Python 3.10+
- Gemini API key

## 3.2 Install

```bash
git clone https://github.com/Nakul-Jaglan/gen-ai-capstone.git
cd gen-ai-capstone

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3.3 Environment Variables

```bash
cp .env.example .env
```

Set in `.env`:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-001
```

## 3.4 Launch

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

---

## 4. Agentic Workflow

LangGraph nodes:
1. **route**: classify query path (greeting / knowledge / pricing)
2. **retrieve**: fetch top-k context chunks using hybrid scoring
3. **pricing**: run RF tool when property payload is available
4. **reason**: generate grounded answer via Gemini 2.5 Flash
5. **guardrail**: confidence estimation + citation append + fallback warnings

ASCII flow:

```text
User Query
   |
   v
[Route Node] ---> [Retrieve Node] ---> [Pricing Tool Node] ---> [Reason Node] ---> [Guardrail Node] ---> Final Answer
```

---

## 5. Evaluation and Results

Use **RAG Diagnostics** tab:
- Retrieval probe for transparency (scores, sources, chunk IDs)
- Benchmark runner over `data/eval_questions.json`

Reported metrics:
- Pass rate
- Average confidence
- Average keyword coverage
- Average latency

This gives qualitative + quantitative performance evidence for the report.

---

## 6. Live Demo (Hosted Link)

Deploy to Streamlit Community Cloud:
1. Push repository to GitHub
2. Go to `share.streamlit.io`
3. Select repo and set entrypoint to `app.py`
4. Add `GEMINI_API_KEY` in app secrets
5. Deploy

Add deployed URL here after publishing:
- `LIVE_DEMO_URL = <your_streamlit_cloud_link>`

---

## 7. API and Failure Handling

- If Gemini generation fails: deterministic fallback response is returned
- If embedding fails: lexical retrieval remains active
- If retrieval is weak: low-confidence warning appears in answer metadata
- If pricing input contains unseen categories: fallback mapping warning is surfaced

---

## 8. Notes for Graders

This repository intentionally demonstrates:
- **GenAI compliance** (not just traditional ML)
- **Agentic orchestration with LangGraph**
- **RAG with retrieval diagnostics and citations**
- **Deployment-ready Streamlit app with API-key-based configuration**
- **Professional project report in LaTeX (`report.tex`)**
