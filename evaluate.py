#!/usr/bin/env python3
"""
evaluate.py — Offline evaluation harness for MediBot.

Runs every query in eval_queries.json through the MediBot retrieval pipeline
(no LLM call by default, use --with-llm to enable LLM scoring) and computes:

  Retrieval metrics (no LLM needed):
    - Hit@1      : correct disease is the top-ranked result
    - Hit@3      : correct disease appears in top 3
    - Hit@5      : correct disease appears in top 5
    - MRR        : Mean Reciprocal Rank
    - Seed found : at least one seed node was matched

  LLM response metrics (requires --with-llm):
    - ROUGE-1 F1
    - ROUGE-2 F1
    - ROUGE-L F1
    - BLEU-1 through BLEU-4

Usage:
    # Retrieval metrics only (fast, no API key needed):
    python evaluate.py

    # Full metrics including ROUGE/BLEU over LLM responses:
    python evaluate.py --with-llm

    # Run only a subset of query types:
    python evaluate.py --types definition symptom

    # Save results to a specific file:
    python evaluate.py --output results/eval_run1.json

    # Run specific query IDs:
    python evaluate.py --ids DEF-001 SYM-001 MIX-001

Dependencies (retrieval-only):
    pip install pandas python-dotenv

Dependencies (with --with-llm):
    pip install rouge-score nltk
    python -c "import nltk; nltk.download('punkt')"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
MEDIBOT = ROOT / "medibot.py"
EVAL_QUERIES = ROOT / "eval_queries.json"
DEFAULT_OUTPUT = ROOT / "eval_results.json"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _norm_name(s: str) -> str:
    """Normalize disease name for comparison — lowercase, strip, unify apostrophes."""
    s = s.lower().strip()
    s = re.sub(r"[\u2018\u2019\u02bc']", "", s)  # remove all apostrophe variants
    return s


def _rank_of(expected: List[str], ranked: List[str]) -> Optional[int]:
    """Return 1-based rank of the first expected disease in ranked list, or None."""
    expected_norm = {_norm_name(e) for e in expected}
    for i, name in enumerate(ranked, start=1):
        if _norm_name(name) in expected_norm:
            return i
    return None


def hit_at_k(expected: List[str], ranked: List[str], k: int) -> bool:
    rank = _rank_of(expected, ranked)
    return rank is not None and rank <= k


def reciprocal_rank(expected: List[str], ranked: List[str]) -> float:
    rank = _rank_of(expected, ranked)
    return 1.0 / rank if rank is not None else 0.0


def rouge_scores(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except ImportError:
        return {"rouge1": -1.0, "rouge2": -1.0, "rougeL": -1.0, "error": "rouge_score not installed"}


def bleu_scores(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute sentence BLEU scores for n=1..4."""
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        try:
            ref_tokens = nltk.word_tokenize(reference.lower())
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            ref_tokens = nltk.word_tokenize(reference.lower())
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())

        sf = SmoothingFunction().method1
        results = {}
        for n in range(1, 5):
            weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
            results[f"bleu{n}"] = sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=sf)
        return results
    except ImportError:
        return {"bleu1": -1.0, "bleu2": -1.0, "bleu3": -1.0, "bleu4": -1.0, "error": "nltk not installed"}


# ---------------------------------------------------------------------------
# MediBot runner
# ---------------------------------------------------------------------------

def _parse_medibot_output(stdout: str) -> Dict[str, Any]:
    """Extract structured fields from medibot.py stdout."""
    lines = stdout.splitlines()

    def section_lines(name: str) -> List[str]:
        out: List[str] = []
        in_sec = False
        marker = f"=== {name} ==="
        for ln in lines:
            if ln.strip() == marker:
                in_sec = True
                continue
            if in_sec and ln.startswith("=== ") and ln.strip().endswith(" ==="):
                break
            if in_sec:
                out.append(ln)
        return out

    # Ranked diseases
    ranked: List[str] = []
    ranked_scores: List[float] = []
    for ln in section_lines("Ranked diseases (graph-traversal score)"):
        m = re.match(r"^(.*?)\s+([0-9]+(?:\.[0-9]+)?)\s*$", ln.rstrip())
        if m:
            ranked.append(m.group(1).strip())
            ranked_scores.append(float(m.group(2)))

    # Encyclopedia-direct hits (from search_encyclopedia_direct)
    enc_direct: List[str] = []
    for ln in section_lines("Encyclopedia-direct hits"):
        m = re.match(r"^(.*?)\s+([0-9]+(?:\.[0-9]+)?)\s*$", ln.rstrip())
        if m:
            name = m.group(1).strip()
            if name not in ranked:
                enc_direct.append(name)

    # Seeds
    seeds: List[str] = []
    seed_block = section_lines("Graph seeds (node_ids)")
    if seed_block:
        seed_line = " ".join(s.strip() for s in seed_block if s.strip())
        if seed_line and not seed_line.startswith("(none"):
            seeds = [x.strip() for x in seed_line.split(",") if x.strip()]

    # Seed trace for definition queries
    seed_graph_nodes_block = section_lines("Seed graph nodes")
    if not seeds:
        for ln in seed_graph_nodes_block:
            v = ln.strip()
            if v:
                seeds.append(v)

    # LLM response
    llm_response = ""
    pos = stdout.find("=== OpenRouter response ===")
    if pos != -1:
        llm_response = stdout[pos + len("=== OpenRouter response ==="):].strip()

    # Matched definition entry
    matched_entry = ""
    m_match = re.search(r"^Matched:\s*(.+)$", stdout, re.MULTILINE)
    if m_match:
        matched_entry = m_match.group(1).strip()

    return {
        "ranked": ranked,
        "ranked_scores": ranked_scores,
        "seeds": seeds,
        "llm_response": llm_response,
        "matched_entry": matched_entry,
        "enc_direct": enc_direct,
    }


def run_medibot(query: str, with_llm: bool = False, timeout: int = 120) -> Dict[str, Any]:
    """Run medibot.py for a single query and return parsed output."""
    cmd = [
        sys.executable, str(MEDIBOT),
        "--dataset-dir", "final_dataset",
        "--user-input", query,
        "--backend", "csv",
    ]
    if not with_llm:
        cmd.append("--no-llm")

    env = os.environ.copy()

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "ranked": [], "seeds": [], "llm_response": "", "matched_entry": ""}
    except Exception as ex:
        return {"error": str(ex), "ranked": [], "seeds": [], "llm_response": "", "matched_entry": ""}

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        return {
            "error": f"returncode={proc.returncode}: {stderr[:200]}",
            "ranked": [],
            "seeds": [],
            "llm_response": "",
            "matched_entry": "",
            "stdout": proc.stdout or "",
        }

    parsed = _parse_medibot_output(proc.stdout or "")
    parsed["stdout"] = proc.stdout or ""
    return parsed


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(
    q: Dict[str, Any],
    with_llm: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    qid = q["id"]
    qtype = q["type"]
    query_text = q["query"]
    expected = q.get("expected_diseases", [])
    reference = q.get("reference_answer", "")

    t0 = time.time()
    result = run_medibot(query_text, with_llm=with_llm)
    elapsed = time.time() - t0

    ranked = result.get("ranked", [])
    enc_direct = result.get("enc_direct", [])
    seeds = result.get("seeds", [])
    llm_response = result.get("llm_response", "")
    matched_entry = result.get("matched_entry", "")
    error = result.get("error", "")

    # For definition queries the "ranked" list may be empty but matched_entry is set
    # — treat matched_entry as the top-1 result for metric computation
    effective_ranked = ranked[:]
    if matched_entry and matched_entry not in effective_ranked:
        effective_ranked.insert(0, matched_entry)
    # Append encyclopedia-direct hits (deduped) after graph results
    for name in enc_direct:
        if name not in effective_ranked:
            effective_ranked.append(name)

    # Retrieval metrics
    metrics: Dict[str, Any] = {
        "qid": qid,
        "type": qtype,
        "query": query_text,
        "expected_diseases": expected,
        "ranked_top5": effective_ranked[:5],
        "seed_count": len(seeds),
        "seed_found": len(seeds) > 0,
        "elapsed_sec": round(elapsed, 2),
        "error": error,
    }

    if expected:
        metrics["hit1"] = hit_at_k(expected, effective_ranked, 1)
        metrics["hit3"] = hit_at_k(expected, effective_ranked, 3)
        metrics["hit5"] = hit_at_k(expected, effective_ranked, 5)
        metrics["mrr"] = reciprocal_rank(expected, effective_ranked)
    else:
        # Edge/negative cases: only check that the system didn't crash
        metrics["hit1"] = None
        metrics["hit3"] = None
        metrics["hit5"] = None
        metrics["mrr"] = None
        metrics["no_crash"] = not bool(error)

    # LLM metrics
    if with_llm and llm_response and reference:
        metrics["llm_response_len"] = len(llm_response.split())
        metrics.update(rouge_scores(llm_response, reference))
        metrics.update(bleu_scores(llm_response, reference))
    elif with_llm:
        metrics["llm_response_len"] = 0
        metrics["rouge1"] = 0.0
        metrics["rouge2"] = 0.0
        metrics["rougeL"] = 0.0
        metrics["bleu1"] = 0.0
        metrics["bleu2"] = 0.0
        metrics["bleu3"] = 0.0
        metrics["bleu4"] = 0.0

    if verbose:
        status = "✓" if (metrics.get("hit1") or metrics.get("no_crash")) else "✗"
        print(f"  [{status}] {qid} | seeds={len(seeds)} | top1={effective_ranked[0] if effective_ranked else 'none'} | {elapsed:.1f}s")
        if error:
            print(f"      ERROR: {error}")

    return metrics


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def aggregate(results: List[Dict[str, Any]], with_llm: bool) -> Dict[str, Any]:
    """Compute macro-averaged metrics across all queries with expected diseases."""
    scored = [r for r in results if r.get("mrr") is not None]
    edge_cases = [r for r in results if r.get("no_crash") is not None]

    def avg(key: str, items=None) -> float:
        items = items or scored
        vals = [r[key] for r in items if r.get(key) is not None and isinstance(r[key], (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    by_type: Dict[str, Any] = {}
    for qtype in ("definition", "symptom", "mixed"):
        subset = [r for r in scored if r["type"] == qtype]
        if subset:
            by_type[qtype] = {
                "n": len(subset),
                "hit1": avg("hit1", subset),
                "hit3": avg("hit3", subset),
                "hit5": avg("hit5", subset),
                "mrr": avg("mrr", subset),
                "seed_found": avg("seed_found", subset),
            }

    summary: Dict[str, Any] = {
        "total_queries": len(results),
        "scored_queries": len(scored),
        "edge_case_queries": len(edge_cases),
        "hit1": avg("hit1"),
        "hit3": avg("hit3"),
        "hit5": avg("hit5"),
        "mrr": avg("mrr"),
        "seed_found_rate": avg("seed_found"),
        "avg_elapsed_sec": avg("elapsed_sec", results),
        "by_type": by_type,
    }

    if with_llm:
        summary["rouge1"] = avg("rouge1", scored)
        summary["rouge2"] = avg("rouge2", scored)
        summary["rougeL"] = avg("rougeL", scored)
        summary["bleu1"] = avg("bleu1", scored)
        summary["bleu2"] = avg("bleu2", scored)
        summary["bleu3"] = avg("bleu3", scored)
        summary["bleu4"] = avg("bleu4", scored)

    crashes = [r for r in edge_cases if not r.get("no_crash", True)]
    summary["edge_crashes"] = len(crashes)
    summary["edge_crash_ids"] = [r["qid"] for r in crashes]

    errors = [r for r in results if r.get("error")]
    summary["errors"] = len(errors)
    summary["error_ids"] = [r["qid"] for r in errors]

    return summary


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(summary: Dict[str, Any], results: List[Dict[str, Any]], with_llm: bool) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  MEDIBOT EVALUATION REPORT")
    print(sep)
    print(f"  Total queries      : {summary['total_queries']}")
    print(f"  Scored queries     : {summary['scored_queries']}")
    print(f"  Edge-case queries  : {summary['edge_case_queries']}")
    print(f"  Errors             : {summary['errors']}")
    if summary.get("error_ids"):
        print(f"  Error IDs          : {', '.join(summary['error_ids'])}")
    print()
    print("  RETRIEVAL METRICS (macro-averaged over scored queries)")
    print(f"  {'Hit@1':<20}: {summary['hit1']:.4f}")
    print(f"  {'Hit@3':<20}: {summary['hit3']:.4f}")
    print(f"  {'Hit@5':<20}: {summary['hit5']:.4f}")
    print(f"  {'MRR':<20}: {summary['mrr']:.4f}")
    print(f"  {'Seed Found Rate':<20}: {summary['seed_found_rate']:.4f}")
    print(f"  {'Avg Latency (s)':<20}: {summary['avg_elapsed_sec']:.2f}")

    if with_llm:
        print()
        print("  LLM RESPONSE METRICS (vs reference answers)")
        print(f"  {'ROUGE-1':<20}: {summary.get('rouge1', 'n/a')}")
        print(f"  {'ROUGE-2':<20}: {summary.get('rouge2', 'n/a')}")
        print(f"  {'ROUGE-L':<20}: {summary.get('rougeL', 'n/a')}")
        print(f"  {'BLEU-1':<20}: {summary.get('bleu1', 'n/a')}")
        print(f"  {'BLEU-2':<20}: {summary.get('bleu2', 'n/a')}")
        print(f"  {'BLEU-3':<20}: {summary.get('bleu3', 'n/a')}")
        print(f"  {'BLEU-4':<20}: {summary.get('bleu4', 'n/a')}")

    if summary.get("by_type"):
        print()
        print("  RETRIEVAL BY QUERY TYPE")
        hdr = f"  {'Type':<12}  {'n':>4}  {'H@1':>6}  {'H@3':>6}  {'H@5':>6}  {'MRR':>6}  {'Seeds':>6}"
        print(hdr)
        print("  " + "-" * 58)
        for qtype, m in summary["by_type"].items():
            print(
                f"  {qtype:<12}  {m['n']:>4}  "
                f"{m['hit1']:>6.3f}  {m['hit3']:>6.3f}  {m['hit5']:>6.3f}  "
                f"{m['mrr']:>6.3f}  {m['seed_found']:>6.3f}"
            )

    # Per-query failures
    failures = [r for r in results if r.get("hit1") is False]
    if failures:
        print()
        print(f"  MISS DETAIL ({len(failures)} queries where Hit@1 = False)")
        for r in failures:
            top1 = r["ranked_top5"][0] if r["ranked_top5"] else "none"
            exp = r["expected_diseases"][:2]
            print(f"  [{r['qid']}] expected={exp} | top1='{top1}' | seeds={r['seed_count']}")

    edge_crashes = summary.get("edge_crashes", 0)
    if edge_crashes:
        print()
        print(f"  EDGE-CASE CRASHES: {edge_crashes}")
        print(f"  IDs: {summary.get('edge_crash_ids', [])}")

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="Evaluate MediBot retrieval and (optionally) LLM response quality."
    )
    ap.add_argument(
        "--queries", default=str(EVAL_QUERIES),
        help="Path to eval_queries.json (default: ./eval_queries.json)"
    )
    ap.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help="Path to write JSON results (default: ./eval_results.json)"
    )
    ap.add_argument(
        "--with-llm", action="store_true",
        help="Also call the LLM and compute ROUGE/BLEU. Requires OPENROUTER_API_KEY."
    )
    ap.add_argument(
        "--types", nargs="+", choices=["definition", "symptom", "mixed", "edge"],
        default=None,
        help="Only evaluate queries of these types."
    )
    ap.add_argument(
        "--ids", nargs="+", default=None,
        help="Only evaluate specific query IDs (e.g. DEF-001 SYM-003)."
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-query result as it completes."
    )
    ap.add_argument(
        "--no-save", action="store_true",
        help="Skip saving results to disk."
    )
    args = ap.parse_args()

    # Load queries
    query_path = Path(args.queries)
    if not query_path.exists():
        print(f"ERROR: eval_queries.json not found at {query_path}", file=sys.stderr)
        sys.exit(1)

    all_queries: List[Dict[str, Any]] = json.loads(query_path.read_text(encoding="utf-8"))

    # Filter
    queries = all_queries
    if args.types:
        queries = [q for q in queries if q["type"] in args.types]
    if args.ids:
        ids_set = set(args.ids)
        queries = [q for q in queries if q["id"] in ids_set]

    if not queries:
        print("No queries matched the given filters.", file=sys.stderr)
        sys.exit(1)

    print(f"\nMediBot Evaluator")
    print(f"  Queries to run : {len(queries)}")
    print(f"  LLM scoring    : {'YES (ROUGE + BLEU)' if args.with_llm else 'NO (retrieval only)'}")
    print(f"  Output file    : {args.output}")
    print()

    if args.with_llm and not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: --with-llm requires OPENROUTER_API_KEY to be set.", file=sys.stderr)
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    for i, q in enumerate(queries, start=1):
        if not args.verbose:
            print(f"  Running {i:>3}/{len(queries)} : {q['id']} ...", end="\r", flush=True)
        else:
            print(f"  [{i:>3}/{len(queries)}] {q['id']}")

        metrics = evaluate_query(q, with_llm=args.with_llm, verbose=args.verbose)
        results.append(metrics)

    if not args.verbose:
        print()  # newline after carriage-return progress

    summary = aggregate(results, with_llm=args.with_llm)
    print_report(summary, results, with_llm=args.with_llm)

    output = {
        "summary": summary,
        "per_query": results,
    }

    if not args.no_save:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Results saved to: {out_path}")

    # Exit non-zero if Hit@1 < 0.5 (useful for CI)
    if summary["hit1"] < 0.5 and summary["scored_queries"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
