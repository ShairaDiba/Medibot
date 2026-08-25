#!/usr/bin/env python3
"""
run_lightrag_pipeline.py — Resume LightRAG indexing, then run full evaluation.

Usage:
    python run_lightrag_pipeline.py
    python run_lightrag_pipeline.py --batch-size 10
    python run_lightrag_pipeline.py --skip-build        # eval only
    python run_lightrag_pipeline.py --skip-eval         # build only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD_SCRIPT = ROOT / "build_lightrag_index.py"
EVAL_SCRIPT = ROOT / "evaluate_lightrag.py"
STATUS_PATH = ROOT / "lightrag_storage" / "kv_store_doc_status.json"
PIPELINE_LOG = ROOT / "lightrag_pipeline.log"

# build_lightrag_index.py exits with this code when the LLM key hits a 402.
EXIT_CREDITS_EXHAUSTED = 42


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def count_processed() -> tuple[int, int, int]:
    if not STATUS_PATH.exists():
        return 0, 0, 0
    data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    processed = sum(1 for meta in data.values() if meta.get("status") == "processed")
    failed = sum(1 for meta in data.values() if meta.get("status") == "failed")
    return processed, failed, len(data)


def expected_total(dataset_dir: Path) -> int:
    from lightrag_utils import load_corpus_documents

    docs, _, _ = load_corpus_documents(dataset_dir)
    return len(docs)


def run_build(batch_size: int, dataset_dir: Path, storage_dir: Path) -> int:
    processed_before, _, _ = count_processed()
    target = expected_total(dataset_dir)
    log(f"Starting/resuming index build ({processed_before}/{target} already processed)")

    cmd = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--resume",
        "--batch-size",
        str(batch_size),
        "--dataset-dir",
        str(dataset_dir),
        "--storage-dir",
        str(storage_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    processed_after, failed, total = count_processed()
    log(f"Build finished with exit={proc.returncode}; processed={processed_after}/{target}, failed={failed}, tracked={total}")
    return proc.returncode


def run_eval(mode: str) -> int:
    log(f"Starting full LightRAG evaluation (mode={mode})")
    cmd = [sys.executable, str(EVAL_SCRIPT), "--mode", mode, "--verbose"]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    log(f"Evaluation finished with exit={proc.returncode}")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume LightRAG build and run full evaluation.")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--dataset-dir", default=str(ROOT / "final_dataset"))
    ap.add_argument("--storage-dir", default=str(ROOT / "lightrag_storage"))
    ap.add_argument("--mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix"])
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    PIPELINE_LOG.write_text("", encoding="utf-8")
    log("LightRAG pipeline started")

    if not args.skip_build:
        build_code = run_build(args.batch_size, Path(args.dataset_dir), Path(args.storage_dir))
        if build_code == EXIT_CREDITS_EXHAUSTED:
            processed, _, _ = count_processed()
            log(f"CREDITS EXHAUSTED at {processed} processed. Pipeline stopped cleanly.")
            log("ACTION NEEDED: put a new OPENROUTER_API_KEY in .env, then re-run this script to resume.")
            sys.exit(EXIT_CREDITS_EXHAUSTED)
        if build_code != 0:
            log("Build failed; skipping evaluation.")
            sys.exit(build_code)

        processed, failed, _ = count_processed()
        target = expected_total(Path(args.dataset_dir))
        if processed < target:
            log(f"WARNING: only {processed}/{target} documents processed ({failed} failed).")
            if failed:
                log("Some documents failed during indexing; evaluation may be incomplete.")

    if not args.skip_eval:
        eval_code = run_eval(args.mode)
        if eval_code != 0:
            sys.exit(eval_code)

    processed, failed, _ = count_processed()
    target = expected_total(Path(args.dataset_dir))
    log(f"Pipeline complete. Index: {processed}/{target} processed ({failed} failed).")
    log(f"Results: {ROOT / 'eval_lightrag_results.json'}")
    log(f"Comparison: {ROOT / 'eval_lightrag_comparison.json'}")


if __name__ == "__main__":
    main()
