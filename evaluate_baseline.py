#!/usr/bin/env python3
"""
evaluate_baseline.py — Unconstrained LLM baseline for MediBot comparison.

Calls the same OpenRouter model (openai/gpt-oss-120b) on every query in
eval_queries.json with NO retrieval context whatsoever — just the raw user
query, exactly as a user would ask a general-purpose LLM. Computes ROUGE and
BLEU against the same reference answers used for MediBot, then produces a
side-by-side comparison table and saves both result files.

This provides the comparison point needed to show that knowledge-graph-grounded
retrieval materially improves response quality over unconstrained generation.

Usage:
    python evaluate_baseline.py                         # runs all 60 queries
    python evaluate_baseline.py --types definition      # one query type only
    python evaluate_baseline.py --ids DEF-001 SYM-001   # specific queries
    python evaluate_baseline.py --dry-run               # print prompts, no API call

Requires:
    OPENROUTER_API_KEY in environment or .env file
    pip install httpx python-dotenv rouge-score nltk
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

# ── paths ──────────────────────────────────────────────────────────────────
ROOT              = Path(__file__).resolve().parent
EVAL_QUERIES      = ROOT / "eval_queries.json"
MEDIBOT_RESULTS   = ROOT / "eval_results.json"
BASELINE_OUTPUT   = ROOT / "eval_baseline_results.json"
COMPARISON_OUTPUT = ROOT / "eval_comparison.json"

# ── model config — same as MediBot ────────────────────────────────────────
DEFAULT_MODEL    = "openai/gpt-oss-120b"
OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE      = 0.25   # identical to MediBot to isolate retrieval effect
MAX_TOKENS       = 1024
TIMEOUT_SEC      = 180

# ── system prompt — no retrieval context, no constraints ──────────────────
BASELINE_SYSTEM = (
    "You are a helpful medical information assistant. "
    "Answer the user's question as clearly and accurately as you can "
    "using your general medical knowledge. "
    "Keep your response concise (under 150 words). "
    "End with: 'This is educational information, not a diagnosis.'"
)


# ── metric helpers ─────────────────────────────────────────────────────────

def rouge_scores(hypothesis: str, reference: str) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 6),
            "rouge2": round(scores["rouge2"].fmeasure, 6),
            "rougeL": round(scores["rougeL"].fmeasure, 6),
        }
    except ImportError:
        return {"rouge1": -1.0, "rouge2": -1.0, "rougeL": -1.0,
                "error": "rouge_score not installed — pip install rouge-score"}


def bleu_scores(hypothesis: str, reference: str) -> Dict[str, float]:
    try:
        import nltk
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        try:
            ref_tokens = nltk.word_tokenize(reference.lower())
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            ref_tokens = nltk.word_tokenize(reference.lower())
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())

        sf = SmoothingFunction().method1
        results: Dict[str, float] = {}
        for n in range(1, 5):
            weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
            results[f"bleu{n}"] = round(
                sentence_bleu(
                    [ref_tokens], hyp_tokens,
                    weights=weights, smoothing_function=sf
                ), 6
            )
        return results
    except ImportError:
        return {"bleu1": -1.0, "bleu2": -1.0, "bleu3": -1.0, "bleu4": -1.0,
                "error": "nltk not installed — pip install nltk"}


# ── OpenRouter call ────────────────────────────────────────────────────────

def call_openrouter(
    query: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Send a bare query to the LLM with no retrieval context."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "X-Title":       "MediBot-Baseline",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user",   "content": query},
        ],
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
    }
    with httpx.Client(timeout=TIMEOUT_SEC) as client:
        resp = client.post(OPENROUTER_URL, json=payload, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter HTTP {resp.status_code}: {resp.text[:400]}"
            )
        data = resp.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as ex:
        raise RuntimeError(f"Unexpected response shape: {data}") from ex


# ── per-query evaluation ───────────────────────────────────────────────────

def evaluate_query(
    q: Dict[str, Any],
    api_key: str,
    model: str,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    qid       = q["id"]
    qtype     = q["type"]
    query_txt = q["query"]
    reference = q.get("reference_answer", "")

    result: Dict[str, Any] = {
        "qid":              qid,
        "type":             qtype,
        "query":            query_txt,
        "expected_diseases": q.get("expected_diseases", []),
        "llm_response":     "",
        "llm_response_len": 0,
        "elapsed_sec":      0.0,
        "error":            "",
        "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
        "bleu1":  0.0, "bleu2":  0.0, "bleu3":  0.0, "bleu4": 0.0,
    }

    if dry_run:
        print(f"\n[DRY-RUN] {qid} | type={qtype}")
        print(f"  SYSTEM: {BASELINE_SYSTEM[:80]}...")
        print(f"  USER:   {query_txt}")
        result["llm_response"] = "[dry-run — no API call made]"
        return result

    t0 = time.time()
    try:
        response = call_openrouter(query_txt, api_key, model)
    except Exception as ex:
        result["error"] = str(ex)
        result["elapsed_sec"] = round(time.time() - t0, 2)
        if verbose:
            print(f"  [{qid}] ERROR: {ex}")
        return result

    elapsed = time.time() - t0
    result["llm_response"]     = response
    result["llm_response_len"] = len(response.split())
    result["elapsed_sec"]      = round(elapsed, 2)

    if reference:
        result.update(rouge_scores(response, reference))
        result.update(bleu_scores(response, reference))

    if verbose:
        print(
            f"  [{qid}] {elapsed:.1f}s | "
            f"ROUGE-1={result['rouge1']:.3f} | "
            f"BLEU-1={result['bleu1']:.3f} | "
            f"words={result['llm_response_len']}"
        )

    return result


# ── aggregate ──────────────────────────────────────────────────────────────

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Edge-case queries have no reference answer — exclude from metric avg
    scored = [r for r in results if r.get("expected_diseases")]

    def avg(key: str, items: Optional[List] = None) -> float:
        items = items or scored
        vals = [r[key] for r in items
                if isinstance(r.get(key), (int, float)) and r[key] >= 0]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    by_type: Dict[str, Any] = {}
    for qtype in ("definition", "symptom", "mixed"):
        subset = [r for r in scored if r["type"] == qtype]
        if subset:
            by_type[qtype] = {
                "n":      len(subset),
                "rouge1": avg("rouge1", subset),
                "rouge2": avg("rouge2", subset),
                "rougeL": avg("rougeL", subset),
                "bleu1":  avg("bleu1",  subset),
                "bleu4":  avg("bleu4",  subset),
                "avg_elapsed_sec": avg("elapsed_sec", subset),
            }

    errors = [r for r in results if r.get("error")]
    return {
        "total_queries":   len(results),
        "scored_queries":  len(scored),
        "model":           DEFAULT_MODEL,
        "temperature":     TEMPERATURE,
        "retrieval":       "none (unconstrained LLM baseline)",
        "rouge1":          avg("rouge1"),
        "rouge2":          avg("rouge2"),
        "rougeL":          avg("rougeL"),
        "bleu1":           avg("bleu1"),
        "bleu2":           avg("bleu2"),
        "bleu3":           avg("bleu3"),
        "bleu4":           avg("bleu4"),
        "avg_elapsed_sec": avg("elapsed_sec", results),
        "by_type":         by_type,
        "errors":          len(errors),
        "error_ids":       [r["qid"] for r in errors],
    }


# ── side-by-side comparison ────────────────────────────────────────────────

def build_comparison(
    baseline_summary: Dict[str, Any],
    medibot_results_path: Path,
) -> Optional[Dict[str, Any]]:
    """Load MediBot eval_results.json and produce a side-by-side table."""
    if not medibot_results_path.exists():
        print(
            f"\n  NOTE: MediBot results not found at {medibot_results_path}. "
            "Run evaluate.py --with-llm first to generate them. "
            "Comparison table will not be saved.",
            file=sys.stderr,
        )
        return None

    medibot_data    = json.loads(medibot_results_path.read_text(encoding="utf-8"))
    medibot_summary = medibot_data["summary"]

    def pct_change(baseline: float, medibot: float) -> str:
        if baseline == 0:
            return "N/A"
        delta = (medibot - baseline) / baseline * 100
        sign  = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f}%"

    metrics = ["rouge1", "rouge2", "rougeL", "bleu1", "bleu2", "bleu3", "bleu4"]
    overall: Dict[str, Any] = {}
    for m in metrics:
        b = baseline_summary.get(m, 0.0)
        mb = medibot_summary.get(m, 0.0)
        overall[m] = {
            "baseline": round(b, 4),
            "medibot":  round(mb, 4),
            "delta":    pct_change(b, mb),
        }

    by_type: Dict[str, Any] = {}
    for qtype in ("definition", "symptom", "mixed"):
        btype  = baseline_summary.get("by_type", {}).get(qtype, {})
        mbtype = medibot_summary.get("by_type", {}).get(qtype, {})
        if btype and mbtype:
            by_type[qtype] = {
                "baseline_rouge1": round(btype.get("rouge1", 0), 4),
                "medibot_rouge1":  round(mbtype.get("hit1",  0), 4),
                "baseline_bleu1":  round(btype.get("bleu1",  0), 4),
            }

    return {
        "description": (
            "Side-by-side comparison of unconstrained LLM baseline "
            "vs MediBot KG-augmented retrieval on the same 60 queries "
            "and reference answers."
        ),
        "model":              DEFAULT_MODEL,
        "medibot_retrieval":  "KG BFS + encyclopedia boost + alias recovery",
        "baseline_retrieval": "none (raw LLM, no context)",
        "overall":            overall,
        "by_type_rouge1":     by_type,
        "medibot_hit1":       medibot_summary.get("hit1"),
        "medibot_mrr":        medibot_summary.get("mrr"),
        "baseline_hit1":      "N/A (no retrieval — metric not applicable)",
        "note": (
            "Hit@1 and MRR are not applicable to the unconstrained baseline "
            "since it performs no ranked retrieval. ROUGE and BLEU scores are "
            "directly comparable as both systems use the same reference answers."
        ),
    }


# ── pretty printer ─────────────────────────────────────────────────────────

def print_report(
    summary: Dict[str, Any],
    comparison: Optional[Dict[str, Any]],
) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print("  UNCONSTRAINED LLM BASELINE — EVALUATION REPORT")
    print(sep)
    print(f"  Model           : {summary['model']}")
    print(f"  Retrieval       : {summary['retrieval']}")
    print(f"  Total queries   : {summary['total_queries']}")
    print(f"  Scored queries  : {summary['scored_queries']}")
    print(f"  Errors          : {summary['errors']}")
    if summary.get("error_ids"):
        print(f"  Error IDs       : {', '.join(summary['error_ids'])}")
    print()
    print("  RESPONSE QUALITY (vs reference answers)")
    for m in ["rouge1", "rouge2", "rougeL", "bleu1", "bleu2", "bleu3", "bleu4"]:
        print(f"  {m:<12}: {summary.get(m, 0):.4f}")
    print(f"  {'Avg latency':<12}: {summary.get('avg_elapsed_sec', 0):.2f}s")

    if summary.get("by_type"):
        print()
        print("  BY QUERY TYPE")
        hdr = f"  {'Type':<12}  {'n':>3}  {'ROUGE-1':>8}  {'ROUGE-2':>8}  {'ROUGE-L':>8}  {'BLEU-1':>8}  {'BLEU-4':>8}"
        print(hdr)
        print("  " + "-" * 60)
        for qtype, m in summary["by_type"].items():
            print(
                f"  {qtype:<12}  {m['n']:>3}  "
                f"{m['rouge1']:>8.4f}  {m['rouge2']:>8.4f}  "
                f"{m['rougeL']:>8.4f}  {m['bleu1']:>8.4f}  {m['bleu4']:>8.4f}"
            )

    if comparison:
        print()
        print(f"  SIDE-BY-SIDE: BASELINE vs MEDIBOT")
        hdr2 = f"  {'Metric':<12}  {'Baseline':>10}  {'MediBot':>10}  {'Change':>10}"
        print(hdr2)
        print("  " + "-" * 48)
        for m, vals in comparison["overall"].items():
            print(
                f"  {m:<12}  {vals['baseline']:>10.4f}  "
                f"{vals['medibot']:>10.4f}  {vals['delta']:>10}"
            )
        print()
        print(f"  MediBot Hit@1   : {comparison['medibot_hit1']}")
        print(f"  MediBot MRR     : {comparison['medibot_mrr']}")
        print(f"  Baseline Hit@1  : {comparison['baseline_hit1']}")

    print(sep)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="Run unconstrained LLM baseline and compare with MediBot."
    )
    ap.add_argument("--queries",  default=str(EVAL_QUERIES),
                    help="Path to eval_queries.json")
    ap.add_argument("--medibot-results", default=str(MEDIBOT_RESULTS),
                    help="Path to MediBot eval_results.json for comparison")
    ap.add_argument("--output",   default=str(BASELINE_OUTPUT),
                    help="Where to save baseline results JSON")
    ap.add_argument("--comparison-output", default=str(COMPARISON_OUTPUT),
                    help="Where to save side-by-side comparison JSON")
    ap.add_argument("--model",    default=DEFAULT_MODEL,
                    help=f"OpenRouter model (default: {DEFAULT_MODEL})")
    ap.add_argument("--types",    nargs="+",
                    choices=["definition", "symptom", "mixed", "edge"],
                    default=None, help="Restrict to these query types")
    ap.add_argument("--ids",      nargs="+", default=None,
                    help="Run only specific query IDs")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print prompts without calling the API")
    ap.add_argument("--verbose",  "-v", action="store_true",
                    help="Print per-query result as it completes")
    ap.add_argument("--no-save",  action="store_true",
                    help="Skip writing output files")
    ap.add_argument("--delay",    type=float, default=1.0,
                    help="Seconds to wait between API calls (default: 1.0)")
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    # Load queries
    query_path = Path(args.queries)
    if not query_path.exists():
        print(f"ERROR: {query_path} not found.", file=sys.stderr)
        sys.exit(1)

    all_queries: List[Dict[str, Any]] = json.loads(
        query_path.read_text(encoding="utf-8")
    )

    # Filter
    queries = all_queries
    if args.types:
        queries = [q for q in queries if q["type"] in args.types]
    if args.ids:
        ids_set = set(args.ids)
        queries = [q for q in queries if q["id"] in ids_set]

    if not queries:
        print("No queries matched filters.", file=sys.stderr)
        sys.exit(1)

    print(f"\nUnconstrained LLM Baseline Evaluator")
    print(f"  Model          : {args.model}")
    print(f"  Queries to run : {len(queries)}")
    print(f"  Dry run        : {args.dry_run}")
    print(f"  Output file    : {args.output}")
    print()

    results: List[Dict[str, Any]] = []
    for i, q in enumerate(queries, start=1):
        if not args.verbose and not args.dry_run:
            print(f"  Running {i:>3}/{len(queries)} : {q['id']} ...",
                  end="\r", flush=True)
        else:
            print(f"  [{i:>3}/{len(queries)}] {q['id']}")

        result = evaluate_query(
            q, api_key, args.model,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        results.append(result)

        # Polite delay between API calls to avoid rate limiting
        if i < len(queries) and not args.dry_run:
            time.sleep(args.delay)

    if not args.verbose and not args.dry_run:
        print()

    summary    = aggregate(results)
    comparison = build_comparison(summary, Path(args.medibot_results))
    print_report(summary, comparison)

    if not args.no_save and not args.dry_run:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "per_query": results},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  Baseline results saved to : {out_path}")

        if comparison:
            comp_path = Path(args.comparison_output)
            comp_path.write_text(
                json.dumps(comparison, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  Comparison table saved to : {comp_path}")


if __name__ == "__main__":
    main()
