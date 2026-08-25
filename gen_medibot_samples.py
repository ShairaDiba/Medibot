"""Regenerate MediBot LLM responses for sample eval queries using the eval key.

Picks a handful of representative queries (definition + symptom + mixed),
runs medibot.py with the EVAL_API_KEY, and saves results to medibot_samples.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Force medibot.py to use the eval key (indexing keys are dead)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
eval_key = os.getenv("EVAL_API_KEY")
if not eval_key:
    raise SystemExit("EVAL_API_KEY not set in .env")
os.environ["OPENROUTER_API_KEY"] = eval_key

SAMPLE_QIDS = ["DEF-001", "DEF-002", "SYM-001", "SYM-002", "MIX-001"]

queries = json.loads((ROOT / "eval_queries.json").read_text(encoding="utf-8"))
by_id = {q["id"]: q for q in queries}

# Reuse the parser from evaluate.py
sys.path.insert(0, str(ROOT))
from evaluate import _parse_medibot_output  # noqa: E402

results = {}
for qid in SAMPLE_QIDS:
    q = by_id[qid]
    query = q["query"]
    print(f"Running {qid}: {query}")
    cmd = [
        sys.executable, str(ROOT / "medibot.py"),
        "--dataset-dir", "final_dataset",
        "--user-input", query,
        "--backend", "csv",
    ]
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )
    if proc.returncode != 0:
        print(f"  ERROR: {proc.stderr[:300]}")
        results[qid] = {"query": query, "error": proc.stderr[:300]}
        continue
    parsed = _parse_medibot_output(proc.stdout)
    results[qid] = {
        "query": query,
        "ranked": parsed["ranked"][:5],
        "llm_response": parsed["llm_response"],
    }
    print(f"  top1={parsed['ranked'][0] if parsed['ranked'] else '?'}  resp_len={len(parsed['llm_response'].split())} words")

out = ROOT / "medibot_samples.json"
out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved -> {out}")
