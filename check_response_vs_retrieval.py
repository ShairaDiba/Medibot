"""Check: does LightRAG's RESPONSE name the correct disease even when retrieval misses?"""
import json
import re
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


def response_mentions_expected(q):
    """Check if any expected disease name appears in the LLM response text."""
    resp = norm(q.get("llm_response", ""))
    # normalize curly apostrophes etc.
    resp = resp.replace("\u2019", "'").replace("\u2018", "'")
    for e in q.get("expected_diseases", []):
        en = norm(e).replace("\u2019", "'").replace("\u2018", "'")
        # word-boundary search to avoid substring false positives
        if re.search(r"\b" + re.escape(en) + r"\b", resp):
            return e
        # also try without possessive/plural nuances: match core tokens
        core = en.rstrip("s")
        if len(core) > 4 and re.search(r"\b" + re.escape(core) + r"\b", resp):
            return e
    return None


misses = [q for q in per if first_hit_rank(q) is None]
hits = [q for q in per if first_hit_rank(q) is not None]

print(f"Total scored: {len(per)}")
print(f"Retrieval hits (top-5): {len(hits)}")
print(f"Retrieval misses (not in top-5): {len(misses)}\n")

# Of the misses, how many still NAME the expected disease in the response?
named = 0
named_list = []
not_named = []
for q in misses:
    found = response_mentions_expected(q)
    if found:
        named += 1
        named_list.append((q["qid"], found, q["llm_response"][:150]))
    else:
        not_named.append((q["qid"], q["query"], q.get("expected_diseases", []), q["llm_response"][:150]))

print(f"Misses where RESPONSE still names the expected disease: {named}/{len(misses)}\n")
print("=== Retrieval MISS but response CORRECT ===")
for qid, found, snippet in named_list:
    print(f"  [{qid}] response mentions '{found}'")
    print(f"        {snippet}...")

print(f"\n=== Retrieval MISS and response also wrong: {len(not_named)}/{len(misses)} ===")
for qid, query, exp, snippet in not_named:
    print(f"  [{qid}] {query[:55]}")
    print(f"        expected: {', '.join(exp)}")
    print(f"        response: {snippet}...")
