import json
import re
from pathlib import Path

from agent import run_agent


FINAL_CONFIG_NAME = "no_planner"
FINAL_CONFIG = {"use_planner": False, "use_reflector": True, "use_verifier": True}
OUTPUT_FILE = Path("predictions.jsonl")
INPUT_FILE = Path("eval/questions.jsonl")


def clean_citations(citation_list):
    """Strips .pdf extensions and ensures a flat, clean list of arXiv IDs."""
    clean_list = []
    for citation in citation_list:
        clean_id = str(citation).replace(".pdf", "").strip()
        match = re.search(r"\d{4}\.\d{4,5}", clean_id)
        if match:
            clean_list.append(match.group(0))
    return list(set(clean_list))


def generate_final_predictions():
    print(f"🚀 Starting final AIMS-DTU Inference using {FINAL_CONFIG_NAME}...")

    with open(INPUT_FILE, "r", encoding="utf-8-sig") as infile, open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue

            data = json.loads(line)
            query_id = data["id"]
            question = data["question"]

            print(f"Processing {query_id}...")

            result = run_agent(question, config=FINAL_CONFIG)

            final_output = {
                "id": query_id,
                "answer": result.get("answer", "No answer generated."),
                "cited_papers": clean_citations(result.get("citations", [])),
            }

            outfile.write(json.dumps(final_output) + "\n")
            outfile.flush()

    print(f"✅ Final submission file generated: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_final_predictions()