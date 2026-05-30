# AIMS-DTU Agentic RAG Ablation Study

**Abstract.** This repository provides a fully local, zero-cost research pipeline that benchmarks agentic RAG scaffolding against a single-shot baseline. It demonstrates that iterative agent loops can improve citation precision and faithfulness, while incurring substantial latency overhead on local 7B-class models—the **Agentic Tax**. All components operate on local hardware with no external API calls.

## Architecture Overview

**1. Memory (Ingestion + Retrieval).**
- **Grobid (Docker)** parses dense academic PDFs into structured XML, preserving citation and section signals beyond raw text splitting.
- **Qdrant Hybrid Search** combines dense vectors (**BAAI/bge-small-en-v1.5**) with lexical sparse vectors (**prithivida/Splade_PP_en_v1**) for robust semantic + keyword retrieval.

**2. State Machine (Deterministic Routing).**
- **Planner -> Retriever -> Reflector -> Synthesizer -> Verifier**
- Implemented as a pure Python, deterministic state machine in `agent.py` to maintain explicit control over routing and failure handling (no DAG framework constraints).
- Runs on **Mistral-7B** via local **Ollama**.

**3. Evaluation (Automated Grading).**
- `evaluate.py` executes 30 questions across five ablations.
- `judge.py` uses LLM-as-a-judge with strict regex extraction to compute precision, recall, and faithfulness.

## Ablation Configurations

- **baseline**: One-shot retrieval + generation.
- **full_agent**: All scaffolding enabled.
- **no_planner**: Direct retrieval query (bypasses LLM query decomposition).
- **no_reflector**: Disables iterative multi-hop retrieval loops.
- **no_verifier**: Skips post-generation citation hallucination checks.

## Setup & Installation (Reproducibility)

```bash
# Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# Local model runtime
ollama run mistral
```

```bash
# Grobid parser (local Docker)
docker run --rm -p 8070:8070 -p 8071:8071 --name grobid lfoppiano/grobid:0.7.3
```

## Execution Pipeline

```bash
python ingest.py          # Builds the Qdrant DB
python evaluate.py        # Runs the ablation state machine
python judge.py           # Calculates precision/recall and faithfulness
```
