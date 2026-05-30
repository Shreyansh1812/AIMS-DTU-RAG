import json
import re
from pathlib import Path

from langchain_community.chat_models import ChatOllama


CONFIGS = [
    "baseline",
    "full_agent",
    "no_planner",
    "no_reflector",
    "no_verifier",
]


def load_jsonl(path: Path, encoding: str = "utf-8"):
    with open(path, "r", encoding=encoding) as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_citations(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def calculate_citation_metrics(predicted_list, expected_list):
    predicted_set = set(normalize_citations(predicted_list))
    expected_set = set(normalize_citations(expected_list))
    intersection = predicted_set & expected_set

    if not predicted_set and not expected_set:
        precision = 1.0
    elif not predicted_set and expected_set:
        precision = 0.0
    else:
        precision = len(intersection) / len(predicted_set)

    if not expected_set:
        recall = 1.0
    else:
        recall = len(intersection) / len(expected_set)

    return precision, recall


def evaluate_answer_quality(question, answer, llm_client):
    prompt = (
        "You are an impartial, strict academic judge. Read the question and the provided "
        "answer. If the answer directly, accurately, and coherently addresses the question, "
        "award a 1. If it is evasive, logically flawed, or obviously hallucinated, award a 0. "
        "You MUST output exactly 'SCORE: 1' or 'SCORE: 0' and nothing else. "
        "\n\nQuestion: {question}\nAnswer: {answer}"
    ).format(question=question, answer=answer)

    response = llm_client.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    match = re.search(r"SCORE:\s*([01])", response_text)
    if match:
        return int(match.group(1))
    return 0


def average(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def main():
    eval_path = Path("eval/questions.jsonl")
    if not eval_path.exists():
        raise FileNotFoundError("Missing eval/questions.jsonl")

    questions = load_jsonl(eval_path, encoding="utf-8-sig")
    questions_by_id = {q["id"]: q for q in questions}

    llm_client = ChatOllama(model="mistral", temperature=0.0, base_url="http://localhost:11434")

    results = []
    for config in CONFIGS:
        pred_path = Path("predictions") / f"{config}.jsonl"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")

        predictions = load_jsonl(pred_path)
        predictions_by_id = {p["id"]: p for p in predictions}

        precision_scores = []
        recall_scores = []
        answer_scores = []
        latencies = []
        tool_calls = []

        for q_id, q in questions_by_id.items():
            pred = predictions_by_id.get(q_id, {})
            predicted_citations = pred.get("citations", [])
            expected_citations = q.get("expected_citations", [])
            precision, recall = calculate_citation_metrics(predicted_citations, expected_citations)

            precision_scores.append(precision)
            recall_scores.append(recall)

            answer = pred.get("answer", "")
            answer_scores.append(evaluate_answer_quality(q["question"], answer, llm_client))

            latency = pred.get("latency_seconds", 0.0) or 0.0
            tool_call_count = pred.get("tool_call_count", 0) or 0
            latencies.append(float(latency))
            tool_calls.append(float(tool_call_count))

        results.append(
            {
                "config": config,
                "citation_precision": average(precision_scores),
                "citation_recall": average(recall_scores),
                "answer_score": average(answer_scores),
                "avg_latency": average(latencies),
                "avg_tool_calls": average(tool_calls),
            }
        )

    print("| Config | Citation Precision | Citation Recall | Answer Score | Avg Latency (s) | Avg Tool Calls |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in results:
        print(
            "| {config} | {precision:.3f} | {recall:.3f} | {answer:.3f} | {latency:.2f} | {tools:.2f} |".format(
                config=row["config"],
                precision=row["citation_precision"],
                recall=row["citation_recall"],
                answer=row["answer_score"],
                latency=row["avg_latency"],
                tools=row["avg_tool_calls"],
            )
        )


if __name__ == "__main__":
    main()
