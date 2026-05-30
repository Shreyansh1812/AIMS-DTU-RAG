#!/usr/bin/env python3
"""
Phase 2: Agentic Loop for a local, zero-cost Deep Research system.

This script orchestrates a cyclic, multi-agent RAG workflow using:
  - Local Ollama (Mistral) via langchain_community.ChatOllama
  - Local Qdrant hybrid collection (dense + sparse) via fastembed

No external services are called; everything stays on localhost.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastembed import SparseTextEmbedding, TextEmbedding
from langchain_community.chat_models import ChatOllama
from qdrant_client import QdrantClient, models

# ---- Configuration constants ----

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"
OLLAMA_TEMPERATURE = 0.1

QDRANT_PATH = Path("./qdrant_arxiv_db")
COLLECTION_NAME = "arxiv_phase1_hybrid"

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"

MAX_SEARCH_ROUNDS = 3
TOP_K = 5
MAX_CONTEXT_CHARS = 12000


def build_llm() -> ChatOllama:
    """Create a local ChatOllama client configured for deterministic reasoning."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
    )


def normalize_sparse_vector(sparse_embedding) -> Dict[str, List[float]]:
    """
    fastembed sparse embeddings may be returned as an object with
    `.indices` and `.values`, or as a tuple/dict. Normalize to a dict.
    """
    if hasattr(sparse_embedding, "indices") and hasattr(sparse_embedding, "values"):
        return {"indices": list(sparse_embedding.indices), "values": list(sparse_embedding.values)}
    if isinstance(sparse_embedding, tuple) and len(sparse_embedding) == 2:
        return {"indices": list(sparse_embedding[0]), "values": list(sparse_embedding[1])}
    if isinstance(sparse_embedding, dict) and "indices" in sparse_embedding and "values" in sparse_embedding:
        return {"indices": list(sparse_embedding["indices"]), "values": list(sparse_embedding["values"])}
    raise TypeError(f"Unsupported sparse embedding format: {type(sparse_embedding)}")


def normalize_dense_vector(dense_embedding) -> List[float]:
    """Ensure dense vectors are plain Python lists."""
    if hasattr(dense_embedding, "tolist"):
        return dense_embedding.tolist()
    return list(dense_embedding)


def _invoke_llm(llm: ChatOllama, prompt: str) -> str:
    """Invoke the local LLM and return a clean string payload."""
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return content.strip()


def _strip_list_prefix(line: str) -> str:
    """Remove common list prefixes like '-', '*', or '1.' from LLM output lines."""
    cleaned = line.strip()
    cleaned = re.sub(r"^[-*•]\s+", "", cleaned)
    cleaned = re.sub(r"^\d+[\).\s]+", "", cleaned)
    return cleaned.strip()


def format_context(chunks: Iterable[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Join context chunks into a prompt-safe string, truncating by character budget.
    """
    joined: List[str] = []
    total = 0
    for chunk in chunks:
        if total + len(chunk) + 2 > max_chars:
            break
        joined.append(chunk)
        total += len(chunk) + 2
    return "\n\n".join(joined)


def run_planner(llm: ChatOllama, query: str) -> List[str]:
    """
    Ask the LLM to decompose the question into 1-3 targeted search queries.
    Output must be clean, newline-separated queries.
    """
    prompt = (
        "You are a research planner. Break the user question into 1-3 focused search queries.\n"
        "Return ONLY the queries, one per line, with no numbering, bullets, or quotes.\n\n"
        f"User question: {query}\n"
    )
    raw = _invoke_llm(llm, prompt)
    lines = [_strip_list_prefix(line) for line in raw.splitlines() if line.strip()]
    return [line for line in lines if line][:3] or [query]


def run_retrieval(
    queries: Iterable[str],
    client: QdrantClient,
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    state: Dict[str, Any],
    seen_context: Optional[set] = None,
    collection_name: str = COLLECTION_NAME,
    limit: int = TOP_K,
) -> None:
    """
    For each query, run hybrid search (dense + sparse) and append context strings
    that include source file, title, section header, and raw text.
    """
    if not client.collection_exists(collection_name):
        raise RuntimeError(f"Qdrant collection not found: {collection_name}")

    if seen_context is None:
        seen_context = set()

    for query in queries:
        dense_vec = normalize_dense_vector(next(dense_model.embed([query])))
        sparse_raw = next(sparse_model.embed([query]))
        sparse_vec = normalize_sparse_vector(sparse_raw)

        if not hasattr(client, "query_points"):
            raise RuntimeError(
                "Hybrid search requires qdrant-client>=1.10.0 with Query API support."
            )

        response = client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"],
                    ),
                    using="sparse",
                    limit=limit,
                ),
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        results = response.points if hasattr(response, "points") else response

        for hit in results:
            payload = hit.payload or {}
            text = payload.get("text", "")
            if not text:
                continue

            source_file = payload.get("source_file", "Unknown Source")
            title = payload.get("paper_title", "Unknown Title")
            section = payload.get("section_header", "Unknown Section")
            context_block = (
                f"Source: {source_file}\nTitle: {title}\nSection: {section}\nText: {text}"
            )
            context_key = f"{source_file}||{title}||{section}||{text}"

            if context_key in seen_context:
                continue
            seen_context.add(context_key)
            state["retrieved_context"].append(context_block)


def run_reflector(llm: ChatOllama, query: str, context: List[str]) -> bool:
    """
    Ask whether the context is sufficient. Return True for YES, False for NO.
    """
    prompt = (
        "You are a reflection agent.\n"
        "Do you have enough information to fully answer the query?\n"
        "Answer ONLY YES or NO.\n\n"
        f"Query: {query}\n\n"
        f"Context:\n{format_context(context)}\n"
    )
    raw = _invoke_llm(llm, prompt)
    return raw.strip().upper() == "YES"


def run_followup_query(llm: ChatOllama, query: str, context: List[str]) -> str:
    """
    Generate one additional, more specific search query based on missing gaps.
    """
    prompt = (
        "You are a query refinement agent.\n"
        "Based on the original question and the context, craft ONE more specific search query\n"
        "to fill missing information. Return a single line with no bullets or numbering.\n\n"
        f"Original question: {query}\n\n"
        f"Context:\n{format_context(context)}\n"
    )
    return _strip_list_prefix(_invoke_llm(llm, prompt))


def run_synthesizer(llm: ChatOllama, query: str, context: List[str]) -> str:
    """
    Draft the final answer using ONLY the provided context and add inline citations
    in the format [Source: XXXXX.pdf].
    """
    prompt = (
        "You are a synthesis agent. Answer the question using ONLY the context below.\n"
        "Attach inline citations to each sentence that uses the context, using the exact "
        "format [Source: XXXXX.pdf].\n"
        "If the context does not contain enough information, say so explicitly.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{format_context(context)}\n"
    )
    return _invoke_llm(llm, prompt)


def run_verifier(llm: ChatOllama, draft_answer: str, context: List[str]) -> str:
    """
    Verify the draft answer against the context and flag unsupported claims.
    """
    prompt = (
        "You are a verification agent. Compare the draft answer to the context.\n"
        "If all claims are supported, respond with ONLY: VERIFIED\n"
        "If there are unsupported claims, respond with ONLY: ISSUES: <list claims>\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Context:\n{format_context(context)}\n"
    )
    return _invoke_llm(llm, prompt)


def run_agent(query: str, config: Dict[str, bool]) -> Dict[str, Any]:
    """
    Orchestrate the agentic loop with runtime ablation toggles.
    Returns a dict with the final answer and optional verification report.
    """
    llm = build_llm()
    client = QdrantClient(path=str(QDRANT_PATH))
    dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    try:
        state: Dict[str, Any] = {
            "original_query": query,
            "sub_queries": [],
            "retrieved_context": [],
            "search_count": 0,
            "ablation_config": {
                "use_planner": bool(config.get("use_planner", False)),
                "use_reflector": bool(config.get("use_reflector", False)),
                "use_verifier": bool(config.get("use_verifier", False)),
            },
        }

        if state["ablation_config"]["use_planner"]:
            state["sub_queries"] = run_planner(llm, query)
        else:
            state["sub_queries"] = [query]

        processed_queries = set()
        seen_context = set()

        while state["search_count"] < MAX_SEARCH_ROUNDS:
            pending_queries = [q for q in state["sub_queries"] if q not in processed_queries]
            if not pending_queries:
                break

            run_retrieval(
                queries=pending_queries,
                client=client,
                dense_model=dense_model,
                sparse_model=sparse_model,
                state=state,
                seen_context=seen_context,
            )
            processed_queries.update(pending_queries)
            state["search_count"] += 1

            if not state["ablation_config"]["use_reflector"]:
                break

            if run_reflector(llm, query, state["retrieved_context"]):
                break

            followup = run_followup_query(llm, query, state["retrieved_context"])
            if followup and followup not in state["sub_queries"]:
                state["sub_queries"].append(followup)
            else:
                break

        draft_answer = run_synthesizer(llm, query, state["retrieved_context"])
        raw_citations = re.findall(r"\[Source:\s*(.*?)\.pdf\]", draft_answer)
        unique_citations = sorted(set(raw_citations))
        verification_report = None

        if state["ablation_config"]["use_verifier"]:
            verification_report = run_verifier(llm, draft_answer, state["retrieved_context"])

        return {
            "answer": draft_answer,
            "verification": verification_report,
            "citations": unique_citations,
            "search_count": state["search_count"],
        }
    finally:
        client.close()


if __name__ == "__main__":
    sample_query = "What are the limitations of OS-Copilot's memory?"

    baseline_config = {"use_planner": False, "use_reflector": False, "use_verifier": False}
    full_agent_config = {"use_planner": True, "use_reflector": True, "use_verifier": True}

    print("=== Baseline (Ablated) ===")
    baseline_result = run_agent(sample_query, baseline_config)
    print(baseline_result["answer"])

    print("\n=== Full Agent ===")
    full_result = run_agent(sample_query, full_agent_config)
    print(full_result["answer"])
    if full_result["verification"]:
        print(f"\nVerification: {full_result['verification']}")
