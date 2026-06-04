# AIMS-DTU: Local Agentic RAG Ablation Study

**Executive Summary.** This repository is a fully local, zero-cost evaluation pipeline that quantifies the **Agentic Tax**: adding agentic routing nodes to a 7B-class local model introduces large latency penalties with minimal accuracy gains due to small-model capability ceilings. The system runs entirely on local hardware via Ollama (Mistral-7B) and a pre-built Qdrant hybrid vector store.

## Architecture & Data Flow

**Pre-built database (ingestion optional).** The vector store in `qdrant_arxiv_db/` is already populated with hybrid dense-sparse embeddings of the paper corpus. If you want to add or refresh papers, run `ingest.py` to rebuild the collection from PDFs in `test_pdfs/`.

**1. Memory Layer (Hybrid Retrieval).**
- **Qdrant Hybrid Search** over **Dense** vectors (BAAI/bge-small-en-v1.5) and **Sparse/Lexical** vectors (prithivida/Splade_PP_en_v1).
- The agent reads directly from the embedded local database directory without any remote API calls.

**2. Agentic State Machine (Deterministic Routing).**
- Implemented in `agent.py` as a **pure Python deterministic state machine** to retain explicit control over routing logic and failure handling (no DAG frameworks like LangGraph).
- Node sequence: **Planner -> Retriever -> Reflector -> Synthesizer -> Verifier**
- Model runtime: **Mistral-7B** via local **Ollama**.

**3. Evaluation Pipeline (Automated Grading).**
- `evaluate.py` runs a final submission pass over `eval/questions.jsonl` and writes `predictions.jsonl` at the repo root.
- `judge.py` performs LLM-as-a-Judge scoring using strict `re.findall` extraction to compute citation precision, recall, and faithfulness without relying on fragile formatting compliance.

## Ablation Configurations

- **baseline**: One-shot retrieval + generation.
- **full_agent**: All scaffolding enabled.
- **no_planner**: Direct retrieval query (bypasses LLM query decomposition).
- **no_reflector**: Disables iterative multi-hop retrieval loops.
- **no_verifier**: Skips post-generation citation hallucination checks.

## Ablation Study Results

| Config | Citation Precision | Citation Recall | Answer Score | Avg Latency (s) | Avg Tool Calls |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.528 | 0.528 | 0.833 | 17.68 | 1.00 |
| full_agent | 0.667 | 0.583 | 0.833 | 58.29 | 2.17 |
| no_planner | 0.667 | 0.667 | 0.833 | 46.88 | 2.17 |
| no_reflector | 0.667 | 0.528 | 0.833 | 28.52 | 1.00 |
| no_verifier | 0.667 | 0.528 | 0.833 | 53.54 | 2.17 |

## Empirical Observations

**The Latency Trap.** The baseline achieves the same qualitative answer score (0.833) in **17.68s**, while the full agent configuration requires **58.29s**, indicating that agentic scaffolding imposes a large compute overhead without accuracy improvements at 7B scale.

**The Planner Liability.** `no_planner` improves citation recall (0.667) over `full_agent` (0.583), suggesting that 7B-scale query decomposition can degrade retrieval quality by producing inferior dense-semantic search vectors.

**Model Homogeneity Bias.** The flatlined answer score (0.833 across all settings) indicates a single 7B model grading its own outputs yields limited discriminative power, masking subtle qualitative differences.

## Installation & Quickstart

```bash
# Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Prerequisite:** Ollama running **mistral** locally on `http://localhost:11434`.

```bash
ollama run mistral
```

**Optional (only if refreshing the corpus):**
```bash
# Start Grobid (Docker) if you need ingestion
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0

# Rebuild the local Qdrant collection from PDFs in test_pdfs/
python ingest.py
```

```bash
python evaluate.py  # Generates predictions.jsonl at repo root
python judge.py     # Parses predictions and outputs metric evaluations
```

## Submission Outputs

- Root submission file: `predictions.jsonl` (one JSON object per line).
- Per-configuration outputs (if generated): `predictions/<config_name>.jsonl`.
