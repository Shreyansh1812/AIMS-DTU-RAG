import json
import time
from pathlib import Path
from agent import run_agent  # Importing the loop you just built

# 1. The Strict Ablation Configurations
CONFIGS = {
    "baseline": {"use_planner": False, "use_reflector": False, "use_verifier": False},
    "full_agent": {"use_planner": True, "use_reflector": True, "use_verifier": True},
    "no_planner": {"use_planner": False, "use_reflector": True, "use_verifier": True},
    "no_reflector": {"use_planner": True, "use_reflector": False, "use_verifier": True},
    "no_verifier": {"use_planner": True, "use_reflector": True, "use_verifier": False}
}

def evaluate():
    eval_file = Path("eval/questions.jsonl")
    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)

    if not eval_file.exists():
        raise FileNotFoundError(f"Missing {eval_file}. Please create the mock data first.")

    # Load the evaluation questions
    with open(eval_file, "r", encoding="utf-8-sig") as f:
        questions = [json.loads(line) for line in f]

    # Run the ablation study
    for config_name, config in CONFIGS.items():
        print(f"\n{'='*40}")
        print(f"Executing Configuration: {config_name.upper()}")
        print(f"{'='*40}")
        
        out_file = out_dir / f"{config_name}.jsonl"
        
        with open(out_file, "w", encoding="utf-8") as f_out:
            for q in questions:
                print(f"Testing Question [{q['id']}]: {q['question']}")
                
                start_time = time.time()
                
                try:
                    # Execute your agent state machine
                    # We use a flexible unpack just in case Codex structured the return differently
                    result = run_agent(q["question"], config)
                    
                    if isinstance(result, dict):
                        answer = result.get("answer", str(result))
                        citations = result.get("citations", [])
                        search_count = result.get("search_count", 1)
                    elif isinstance(result, tuple) and len(result) >= 3:
                        answer, citations, search_count = result[:3]
                    else:
                        answer = str(result)
                        citations = []
                        search_count = 1
                        
                except Exception as e:
                    print(f"ERROR: Agent crashed on {q['id']} - {e}")
                    answer = f"ERROR: {e}"
                    citations = []
                    search_count = 0

                latency = round(time.time() - start_time, 2)
                
                # Construct the AIMS-DTU submission format
                output_data = {
                    "id": q["id"],
                    "answer": answer,
                    "citations": citations,
                    "latency_seconds": latency,
                    "tool_call_count": search_count
                }
                
                f_out.write(json.dumps(output_data) + "\n")
                print(f" -> Completed in {latency}s | Tool Calls: {search_count}\n")

if __name__ == "__main__":
    evaluate()