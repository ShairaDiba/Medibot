#!/usr/bin/env python3
"""
evaluate_lightrag.py — LightRAG baseline evaluation for MediBot comparison.

Runs every query in eval_queries.json through a LightRAG index built from the
same encyclopedia corpus, then scores retrieval (Hit@1/3/5, MRR) and response
quality (ROUGE/BLEU) against the reference answers. Produces a 3-way comparison
with MediBot and the unconstrained LLM baseline.

Usage:
    python build_lightrag_index.py                 # first-time index build
    python evaluate_lightrag.py                    # all 60 queries
    python evaluate_lightrag.py --ids DEF-001 SYM-001
    python evaluate_lightrag.py --mode hybrid
    python evaluate_lightrag.py --dry-run

Requires:
    OPENROUTER_API_KEY in .env
    pip install lightrag-hku httpx python-dotenv rouge-score nltk
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from evaluate import hit_at_k, reciprocal_rank
from lightrag_utils import (
    DEFAULT_DATASET_DIR,
    DEFAULT_STORAGE_DIR,
    create_lightrag,
    load_encyclopedia_name_index,
    query_lightrag_full,
    storage_is_ready,
)

ROOT = Path(__file__).resolve().parent
EVAL_QUERIES = ROOT / "eval_queries.json"
MEDIBOT_RESULTS = ROOT / "eval_results.json"
BASELINE_RESULTS = ROOT / "eval_baseline_results.json"
LIGHTRAG_OUTPUT = ROOT / "eval_lightrag_results.json"
COMPARISON_OUTPUT = ROOT / "eval_lightrag_comparison.json"

DEFAULT_MODEL = "openai/gpt-oss-120b"


def rouge_scores(hypothesis: str, reference: str) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 6),
            "rouge2": round(scores["rouge2"].fmeasure, 6),
            "rougeL": round(scores["rougeL"].fmeasure, 6),
        }
    except ImportError:
        return {
            "rouge1": -1.0,
            "rouge2": -1.0,
            "rougeL": -1.0,
            "error": "rouge_score not installed — pip install rouge-score",
        }


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
                sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=sf),
                6,
            )
        return results
    except ImportError:
        return {
            "bleu1": -1.0,
            "bleu2": -1.0,
            "bleu3": -1.0,
            "bleu4": -1.0,
            "error": "nltk not installed — pip install nltk",
        }


async def evaluate_query(
    rag,
    q: Dict[str, Any],
    mode: str,
    norm_to_name: Dict[str, str],
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    qid = q["id"]
    qtype = q["type"]
    query_txt = q["query"]
    reference = q.get("reference_answer", "")
    expected = q.get("expected_diseases", [])

    result: Dict[str, Any] = {
        "qid": qid,
        "type": qtype,
        "query": query_txt,
        "expected_diseases": expected,
        "ranked_top5": [],
        "llm_response": "",
        "llm_response_len": 0,
        "elapsed_sec": 0.0,
        "error": "",
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0,
    }

    if dry_run:
        result["llm_response"] = "[dry-run — no LightRAG query made]"
        return result

    t0 = time.time()
    try:
        payload = await query_lightrag_full(
            rag,
            query_txt,
            mode=mode,
            norm_to_name=norm_to_name,
        )
    except Exception as ex:
        result["error"] = str(ex)
        result["elapsed_sec"] = round(time.time() - t0, 2)
        if verbose:
            print(f"  [{qid}] ERROR: {ex}")
        return result

    elapsed = time.time() - t0
    response = payload["response"]
    ranked = payload["ranked"]
    result["llm_response"] = response
    result["llm_response_len"] = len(response.split())
    result["ranked_top5"] = ranked[:5]
    result["elapsed_sec"] = round(elapsed, 2)

    if expected:
        result["hit1"] = hit_at_k(expected, ranked, 1)
        result["hit3"] = hit_at_k(expected, ranked, 3)
        result["hit5"] = hit_at_k(expected, ranked, 5)
        result["mrr"] = reciprocal_rank(expected, ranked)
    else:
        result["hit1"] = None
        result["hit3"] = None
        result["hit5"] = None
        result["mrr"] = None
        result["no_crash"] = not bool(result["error"])

    if reference:
        result.update(rouge_scores(response, reference))
        result.update(bleu_scores(response, reference))

    if verbose:
        top1 = ranked[0] if ranked else "none"
        hit_flag = result.get("hit1")
        print(
            f"  [{qid}] {elapsed:.1f}s | Hit@1={hit_flag} | top1={top1} | "
            f"ROUGE-1={result['rouge1']:.3f} | BLEU-1={result['bleu1']:.3f}"
        )

    return result


def aggregate(results: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    scored = [r for r in results if r.get("expected_diseases")]
    edge_cases = [r for r in results if r.get("no_crash") is not None]

    def avg(key: str, items: Optional[List] = None) -> float:
        items = items or scored
        vals = [
            r[key]
            for r in items
            if r.get(key) is not None and isinstance(r[key], (int, float)) and r[key] >= 0
        ]
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
                "rouge1": avg("rouge1", subset),
                "rouge2": avg("rouge2", subset),
                "rougeL": avg("rougeL", subset),
                "bleu1": avg("bleu1", subset),
                "bleu4": avg("bleu4", subset),
                "avg_elapsed_sec": avg("elapsed_sec", subset),
            }

    errors = [r for r in results if r.get("error")]
    return {
        "total_queries": len(results),
        "scored_queries": len(scored),
        "edge_case_queries": len(edge_cases),
        "model": DEFAULT_MODEL,
        "retrieval": f"LightRAG ({mode} mode, encyclopedia corpus)",
        "hit1": avg("hit1"),
        "hit3": avg("hit3"),
        "hit5": avg("hit5"),
        "mrr": avg("mrr"),
        "rouge1": avg("rouge1"),
        "rouge2": avg("rouge2"),
        "rougeL": avg("rougeL"),
        "bleu1": avg("bleu1"),
        "bleu2": avg("bleu2"),
        "bleu3": avg("bleu3"),
        "bleu4": avg("bleu4"),
        "avg_elapsed_sec": avg("elapsed_sec", results),
        "by_type": by_type,
        "errors": len(errors),
        "error_ids": [r["qid"] for r in errors],
    }


def pct_change(base: float, other: float) -> str:
    if base == 0:
        return "N/A"
    delta = (other - base) / base * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def build_comparison(
    lightrag_summary: Dict[str, Any],
    medibot_path: Path,
    baseline_path: Path,
) -> Dict[str, Any]:
    systems: Dict[str, Dict[str, Any]] = {"lightrag": lightrag_summary}

    if medibot_path.exists():
        medibot_data = json.loads(medibot_path.read_text(encoding="utf-8"))
        systems["medibot"] = medibot_data["summary"]
    if baseline_path.exists():
        baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
        systems["baseline"] = baseline_data["summary"]

    metrics = ["rouge1", "rouge2", "rougeL", "bleu1", "bleu2", "bleu3", "bleu4"]
    overall: Dict[str, Any] = {}
    for metric in metrics:
        row = {name: round(summary.get(metric, 0.0), 4) for name, summary in systems.items()}
        if "baseline" in row and "lightrag" in row:
            row["lightrag_vs_baseline"] = pct_change(row["baseline"], row["lightrag"])
        if "medibot" in row and "lightrag" in row:
            row["medibot_vs_lightrag"] = pct_change(row["lightrag"], row["medibot"])
        overall[metric] = row

    comparison = {
        "description": (
            "Three-way comparison: unconstrained LLM baseline vs LightRAG vs MediBot "
            "on the same eval_queries.json reference answers."
        ),
        "systems": {
            "baseline": systems.get("baseline", {}).get("retrieval", "missing"),
            "lightrag": systems.get("lightrag", {}).get("retrieval", "missing"),
            "medibot": "KG BFS + encyclopedia boost + alias recovery",
        },
        "overall": overall,
        "retrieval_metrics": {
            metric: {
                "baseline": "N/A",
                "lightrag": round(systems.get("lightrag", {}).get(metric, 0.0), 4),
                "medibot": round(systems.get("medibot", {}).get(metric, 0.0), 4),
                "medibot_vs_lightrag": pct_change(
                    systems.get("lightrag", {}).get(metric, 0.0),
                    systems.get("medibot", {}).get(metric, 0.0),
                ),
            }
            for metric in ("hit1", "hit3", "hit5", "mrr")
            if systems.get("medibot")
        },
        "retrieval_by_type": _retrieval_by_type(systems),
        "note": (
            "Hit@K and MRR for LightRAG are computed from retrieved chunk Entry: lines "
            "and disease-like entities mapped to encyclopedia entry names. "
            "The unconstrained baseline performs no retrieval (N/A). "
            "ROUGE/BLEU are directly comparable across all three LLM-backed systems."
        ),
    }
    return comparison


def _retrieval_by_type(systems: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    medibot_types = systems.get("medibot", {}).get("by_type", {})
    lightrag_types = systems.get("lightrag", {}).get("by_type", {})
    out: Dict[str, Any] = {}
    for qtype in ("definition", "symptom", "mixed"):
        if qtype in medibot_types or qtype in lightrag_types:
            out[qtype] = {
                "lightrag_hit1": round(lightrag_types.get(qtype, {}).get("hit1", 0.0), 4),
                "medibot_hit1": round(medibot_types.get(qtype, {}).get("hit1", 0.0), 4),
                "lightrag_mrr": round(lightrag_types.get(qtype, {}).get("mrr", 0.0), 4),
                "medibot_mrr": round(medibot_types.get(qtype, {}).get("mrr", 0.0), 4),
            }
    return out


def print_report(summary: Dict[str, Any], comparison: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  LIGHTRAG BASELINE — EVALUATION REPORT")
    print(sep)
    print(f"  Retrieval       : {summary['retrieval']}")
    print(f"  Total queries   : {summary['total_queries']}")
    print(f"  Scored queries  : {summary['scored_queries']}")
    print(f"  Errors          : {summary['errors']}")
    if summary.get("error_ids"):
        print(f"  Error IDs       : {', '.join(summary['error_ids'])}")
    print()
    print("  RETRIEVAL METRICS (macro-averaged over scored queries)")
    print(f"  {'Hit@1':<12}: {summary.get('hit1', 0):.4f}")
    print(f"  {'Hit@3':<12}: {summary.get('hit3', 0):.4f}")
    print(f"  {'Hit@5':<12}: {summary.get('hit5', 0):.4f}")
    print(f"  {'MRR':<12}: {summary.get('mrr', 0):.4f}")
    print()
    print("  RESPONSE QUALITY (vs reference answers)")
    for metric in ["rouge1", "rouge2", "rougeL", "bleu1", "bleu2", "bleu3", "bleu4"]:
        print(f"  {metric:<12}: {summary.get(metric, 0):.4f}")
    print(f"  {'Avg latency':<12}: {summary.get('avg_elapsed_sec', 0):.2f}s")

    if summary.get("by_type"):
        print()
        print("  RETRIEVAL BY QUERY TYPE")
        print(f"  {'Type':<12}  {'n':>3}  {'H@1':>6}  {'H@3':>6}  {'MRR':>6}")
        print("  " + "-" * 40)
        for qtype, metrics in summary["by_type"].items():
            print(
                f"  {qtype:<12}  {metrics['n']:>3}  "
                f"{metrics.get('hit1', 0):>6.3f}  {metrics.get('hit3', 0):>6.3f}  "
                f"{metrics.get('mrr', 0):>6.3f}"
            )
        print()
        print("  RESPONSE QUALITY BY QUERY TYPE")
        print(f"  {'Type':<12}  {'n':>3}  {'ROUGE-1':>8}  {'ROUGE-L':>8}  {'BLEU-1':>8}  {'BLEU-4':>8}")
        print("  " + "-" * 58)
        for qtype, metrics in summary["by_type"].items():
            print(
                f"  {qtype:<12}  {metrics['n']:>3}  "
                f"{metrics['rouge1']:>8.4f}  {metrics['rougeL']:>8.4f}  "
                f"{metrics['bleu1']:>8.4f}  {metrics['bleu4']:>8.4f}"
            )

    print()
    print("  THREE-WAY COMPARISON")
    print(f"  {'Metric':<10}  {'Baseline':>10}  {'LightRAG':>10}  {'MediBot':>10}")
    print("  " + "-" * 48)
    for metric in ("rouge1", "bleu1"):
        row = comparison["overall"].get(metric, {})
        print(
            f"  {metric:<10}  "
            f"{row.get('baseline', 0):>10.4f}  "
            f"{row.get('lightrag', 0):>10.4f}  "
            f"{row.get('medibot', 0):>10.4f}"
        )

    retrieval = comparison.get("retrieval_metrics", {})
    if retrieval:
        print()
        print("  RETRIEVAL COMPARISON (LightRAG vs MediBot)")
        print(f"  {'Metric':<10}  {'LightRAG':>10}  {'MediBot':>10}  {'Delta':>10}")
        print("  " + "-" * 48)
        for metric in ("hit1", "hit3", "hit5", "mrr"):
            row = retrieval.get(metric, {})
            print(
                f"  {metric:<10}  "
                f"{row.get('lightrag', 0):>10.4f}  "
                f"{row.get('medibot', 0):>10.4f}  "
                f"{row.get('medibot_vs_lightrag', 'N/A'):>10}"
            )

    failures = [r for r in results if r.get("hit1") is False]
    if failures:
        print()
        print(f"  LIGHTRAG MISS DETAIL ({len(failures)} queries where Hit@1 = False)")
        for r in failures[:15]:
            top1 = r["ranked_top5"][0] if r.get("ranked_top5") else "none"
            exp = r.get("expected_diseases", [])[:2]
            print(f"  [{r['qid']}] expected={exp} | top1='{top1}'")

    print(sep)


async def run_eval(args: argparse.Namespace) -> None:
    if not args.dry_run and not storage_is_ready(Path(args.storage_dir)):
        print(
            f"ERROR: LightRAG index not found at {args.storage_dir}.\n"
            "Run: python build_lightrag_index.py",
            file=sys.stderr,
        )
        sys.exit(1)

    query_path = Path(args.queries)
    all_queries: List[Dict[str, Any]] = json.loads(query_path.read_text(encoding="utf-8"))

    queries = all_queries
    if args.types:
        queries = [q for q in queries if q["type"] in args.types]
    if args.ids:
        wanted = set(args.ids)
        queries = [q for q in queries if q["id"] in wanted]
    if not queries:
        print("No queries matched filters.", file=sys.stderr)
        sys.exit(1)

    print("\nLightRAG Baseline Evaluator")
    print(f"  Mode           : {args.mode}")
    print(f"  Queries to run : {len(queries)}")
    print(f"  Storage dir    : {args.storage_dir}")
    print(f"  Dry run        : {args.dry_run}")
    print()

    _, norm_to_name = load_encyclopedia_name_index(Path(args.dataset_dir))

    rag = None
    if not args.dry_run:
        rag = await create_lightrag(storage_dir=Path(args.storage_dir), use_eval_key=True)

    results: List[Dict[str, Any]] = []
    try:
        for i, q in enumerate(queries, start=1):
            if not args.verbose and not args.dry_run:
                print(f"  Running {i:>3}/{len(queries)} : {q['id']} ...", end="\r", flush=True)
            elif args.verbose or args.dry_run:
                print(f"  [{i:>3}/{len(queries)}] {q['id']}")

            result = await evaluate_query(
                rag,
                q,
                mode=args.mode,
                norm_to_name=norm_to_name,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
            results.append(result)

            if i < len(queries) and not args.dry_run and args.delay > 0:
                await asyncio.sleep(args.delay)
    finally:
        if rag is not None:
            await rag.finalize_storages()

    if not args.verbose and not args.dry_run:
        print()

    summary = aggregate(results, mode=args.mode)
    comparison = build_comparison(
        summary,
        Path(args.medibot_results),
        Path(args.baseline_results),
    )
    print_report(summary, comparison, results)

    if not args.no_save and not args.dry_run:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps({"summary": summary, "per_query": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        comp_path = Path(args.comparison_output)
        comp_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  LightRAG results saved to : {out_path}")
        print(f"  Comparison table saved to : {comp_path}")


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Evaluate LightRAG baseline against MediBot.")
    ap.add_argument("--queries", default=str(EVAL_QUERIES))
    ap.add_argument("--storage-dir", default=str(DEFAULT_STORAGE_DIR))
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--medibot-results", default=str(MEDIBOT_RESULTS))
    ap.add_argument("--baseline-results", default=str(BASELINE_RESULTS))
    ap.add_argument("--output", default=str(LIGHTRAG_OUTPUT))
    ap.add_argument("--comparison-output", default=str(COMPARISON_OUTPUT))
    ap.add_argument(
        "--mode",
        default="mix",
        choices=["local", "global", "hybrid", "naive", "mix"],
        help="LightRAG retrieval mode (default: mix = KG + vector).",
    )
    ap.add_argument("--types", nargs="+", choices=["definition", "symptom", "mixed", "edge"], default=None)
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--delay", type=float, default=0.5, help="Delay between queries (seconds).")
    args = ap.parse_args()

    try:
        asyncio.run(run_eval(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
