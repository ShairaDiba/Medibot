"""Analyze LightRAG Hit@k curve and first-hit rank distribution."""
import json
from collections import Counter
from pathlib import Path

data = json.loads(Path("eval_lightrag_results.json").read_text(encoding="utf-8"))
per = [q for q in data["per_query"] if q.get("type") != "edge"]


def norm(s):
    return " ".join(str(s).lower().split())


def first_hit_rank(q):
    exp = {norm(e) for e in q.get("expected_diseases", [])}
    for i, t in enumerate(q.get("ranked_top5", []), 1):
        if norm(t) in exp:
            return i
    return None


n = len(per)
print(f"Scored queries: {n}\n")

# Hit@k curve (ranked_top5 stores only top 5)
print("Hit@k curve (cumulative):")
print(f"{'k':>4} {'Hits':>6} {'Hit@k':>8}")
for k in range(1, 6):
    hits = sum(1 for q in per if (first_hit_rank(q) or 99) <= k)
    print(f"{k:>4} {hits:>6} {hits / n:>8.4f}")

# First-hit rank distribution
print("\nFirst-hit rank distribution:")
c = Counter()
for q in per:
    r = first_hit_rank(q)
    c[r if r else "miss (not in top-5)"] += 1
for r in [1, 2, 3, 4, 5, "miss (not in top-5)"]:
    cnt = c.get(r, 0)
    bar = "#" * cnt
    print(f"  rank {str(r):>18}: {cnt:>3} ({cnt / n * 100:5.1f}%) {bar}")

# Per-type Hit@k
print("\nHit@k by query type:")
for qtype in ["definition", "symptom", "mixed"]:
    sub = [q for q in per if q.get("type") == qtype]
    m = len(sub)
    row = []
    for k in range(1, 6):
        hits = sum(1 for q in sub if (first_hit_rank(q) or 99) <= k)
        row.append(f"{hits / m:.3f}")
    print(f"  {qtype:>12} (n={m:>2}):  H@1={row[0]}  H@2={row[1]}  H@3={row[2]}  H@4={row[3]}  H@5={row[4]}")

# List queries that never hit in top-5
never = [(q["qid"], q["query"], q.get("expected_diseases", []), q.get("ranked_top5", []))
         for q in per if first_hit_rank(q) is None]
print(f"\nNever found in top-5: {len(never)}/{n}")
for qid, query, exp, top5 in never:
    print(f"  [{qid}] {query[:60]}")
    print(f"         expected: {', '.join(exp)}")
    print(f"         got top5: {', '.join(top5)}")
