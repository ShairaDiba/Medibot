#!/usr/bin/env python3
"""
MediBot: knowledge-graph-first medical assistant.

Uses final_dataset:
  02_knowledge_graph_nodes.csv — entity catalogue
  03_knowledge_graph_edges.csv — AFFECTS, HAS_SYMPTOM, TREATS (TREATS is weak; still shown with caveat)
  01_medical_encyclopedia_entries.csv — full article text keyed by entry_name (= graph Disease label)
  04_disease_symptom_matrix.csv — optional binary row snippet for matched diseases
  05_medical_glossary.csv — term -> related entry names for extra seeds

Flow: text -> seed nodes (labels + glossary + symptom-alias vocabulary) -> BFS on graph
-> rank Disease nodes -> pull encyclopedia + local edges -> LLM reasons only over that bundle.

Provider: openrouter (GPT OSS 120B by default). Backend for graph traversal: csv (default), neo4j.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import pandas as pd
from dotenv import load_dotenv

ENCYCLOPEDIA_TEXT_COLUMNS = [
    "icd_category",
    "body_systems",
    "severity_level",
    "age_groups_affected",
    "inheritance_pattern",
    "is_contagious",
    "definition",
    "description",
    "causes",
    "symptoms",
    "diagnosis",
    "treatment",
    "prognosis",
    "prevention",
    "complications",
    "key_terms",
]

REJECT_ENTRY_NAME = [
    r"\bsee\b",
    r"^G AL E",
    r"^KEY TERMS\b",
    r"^BOOKS\b",
    r"^PERIODICALS\b",
    r"Table by",
    r"Cengage Learning",
    r"PreMediaGlobal",
    r"^\(",
]


def is_valid_graph_label(name: str) -> bool:
    """Stricter than encyclopedia entry names: graph Disease labels include many OCR/sentence fragments."""
    if not is_valid_entry_name(name):
        return False
    s = name.strip()
    if len(s) > 72:
        return False
    if s.count(" ") > 10:
        return False
    if s.endswith(".") and s.count(" ") >= 4:
        return False
    low = s.lower()
    if ", and " in low:
        return False
    if re.search(r"\b(pauses|awakenings|sudden)\b.*\b(and|,)\b", low):
        return False
    if re.search(r"\b(when|which|that)\s+(he|she|they)\s+", low):
        return False
    if re.search(r"\bhim when\b|\bher when\b", low):
        return False
    if re.search(r"\b\d{1,2},\s*\d{4}\)", low):
        return False
    return True


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "i", "me", "my", "you", "your",
    "what", "which", "who", "when", "where", "why", "how", "and", "but", "or",
    "if", "with", "from", "into", "about", "this", "that", "for", "to", "of",
    "in", "on", "at", "by", "not", "no", "yes", "get", "got", "like", "just",
    "tell", "give", "any", "some", "all", "more", "very",
}

DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b"

OPENROUTER_BASE_URL_DEFAULT = "https://openrouter.ai/api/v1"

RESPIRATORY_EXPERIENCE_TOKENS = frozenset(
    {"breathing", "breath", "breathe", "gasping", "gasp", "wheeze", "wheezing", "wheezes"}
)

MAX_BFS_NODES = 200
TOP_DISEASES = 12
TOP_CANDIDATE_EXCERPTS = 8
EDGE_LINES_PER_DISEASE = 25


def clean_text(v: object) -> str:
    if pd.isna(v):
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def norm_user(s: str) -> str:
    return clean_text(s).lower()


def is_valid_entry_name(name: str) -> bool:
    if not name or len(name) < 3:
        return False
    if name.strip().lower().startswith("or "):
        return False
    if name.strip().lower() == "intent":
        return False
    for p in REJECT_ENTRY_NAME:
        if re.search(p, name, flags=re.I):
            return False
    return True


def user_word_tokens(text: str) -> List[str]:
    low = norm_user(text)
    return [t for t in re.findall(r"[a-z0-9]+", low) if t not in STOPWORDS and len(t) >= 2]


def parse_definition_query(user_text: str) -> Optional[str]:
    low = norm_user(user_text)
    patterns = [
        r"^what\s+is\s+(.+?)\??$",
        r"^what\s+are\s+(.+?)\??$",
        r"^define\s+(.+?)\??$",
        r"^explain\s+(.+?)\??$",
        r"^(?:tell\s+me\s+about)\s+(.+?)\??$",
        r"^(?:information\s+on|info\s+on|about)\s+(.+?)\??$",
    ]
    for p in patterns:
        m = re.match(p, low)
        if m:
            topic = m.group(1).strip().strip('"').strip("'")
            if len(topic) >= 2:
                return topic
    return None


def find_encyclopedia_row(entries: pd.DataFrame, topic: str) -> Optional[pd.Series]:
    t_low = topic.strip().lower()
    names = entries["entry_name"].astype(str)
    exact = entries[names.str.lower() == t_low]
    if len(exact) == 1:
        return exact.iloc[0]
    if len(exact) > 1:
        dis = exact[exact["entry_type"].astype(str).str.contains("Disease", na=False)]
        return dis.iloc[0] if len(dis) else exact.iloc[0]
    dis_only = entries[entries["entry_type"].astype(str).str.contains("Disease", na=False, case=False)]
    starts = dis_only[dis_only["entry_name"].astype(str).str.lower().str.startswith(t_low, na=False)]
    if len(starts):
        return starts.loc[starts["entry_name"].astype(str).str.len().idxmin()]
    contains = dis_only[
        dis_only["entry_name"].astype(str).str.lower().str.contains(re.escape(t_low), na=False)
    ]
    if len(contains):
        return contains.loc[contains["entry_name"].astype(str).str.len().idxmin()]
    return None


def format_encyclopedia_context(row: pd.Series, max_field: int = 1000) -> str:
    lines = [f"entry_name: {row.get('entry_name', '')}", f"entry_type: {row.get('entry_type', '')}"]
    for col in ENCYCLOPEDIA_TEXT_COLUMNS:
        if col not in row.index:
            continue
        val = row.get(col)
        if pd.isna(val) or str(val).strip() == "":
            continue
        text = clean_text(val)
        if len(text) > max_field:
            text = text[:max_field] + "..."
        lines.append(f"{col}: {text}")
    return "\n".join(lines)


def load_symptom_alias_vocabulary(dataset_dir: Path) -> Dict[str, Set[str]]:
    """symptom_aliases.json is derived from the matrix symptom columns (dataset-native vocabulary)."""
    p = dataset_dir / "symptom_aliases.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: {str(x).lower() for x in v} for k, v in raw.items()}


def build_graph_indices(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> Tuple[Dict[str, dict], DefaultDict[str, List[str]], DefaultDict[str, List[Tuple[str, str, str, str]]]]:
    """node_id -> record; label_lower -> node_ids; undirected adjacency id -> list of (neighbor_id, rel, src_l, tgt_l)."""
    by_id: Dict[str, dict] = {}
    label_index: DefaultDict[str, List[str]] = defaultdict(list)
    for _, r in nodes.iterrows():
        nid = str(r["node_id"])
        by_id[nid] = {
            "node_id": nid,
            "node_type": str(r.get("node_type", "")),
            "label": str(r.get("label", "")),
            "definition": str(r.get("definition", "")) if pd.notna(r.get("definition")) else "",
        }
        lb = str(r.get("label", "")).strip().lower()
        if lb:
            label_index[lb].append(nid)

    adj: DefaultDict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
    for _, e in edges.iterrows():
        rel = str(e["relationship"])
        sid = str(e["source_node_id"])
        tid = str(e["target_node_id"])
        sl = str(e["source_label"])
        tl = str(e["target_label"])
        adj[sid].append((tid, rel, sl, tl))
        adj[tid].append((sid, rel, sl, tl))
    return by_id, label_index, adj


def label_to_node_ids(label: str, label_index: DefaultDict[str, List[str]], nodes: pd.DataFrame) -> List[str]:
    key = label.strip().lower()
    if key in label_index:
        return list(label_index[key])
    hits: List[str] = []
    for _, r in nodes.iterrows():
        if str(r.get("label", "")).strip().lower() == key:
            hits.append(str(r["node_id"]))
    return hits


def match_seed_nodes(
    user_text: str,
    nodes: pd.DataFrame,
    symptom_vocab: Dict[str, Set[str]],
) -> Tuple[Set[str], List[str]]:
    """Return seed node_ids and human-readable seed reasons."""
    seeds: Set[str] = set()
    reasons: List[str] = []
    low = norm_user(user_text)
    tokens = user_word_tokens(user_text)

    def add_id(nid: str, why: str) -> None:
        if nid and nid not in seeds:
            seeds.add(nid)
            reasons.append(f"{why} -> {nid}")

    for _, r in nodes.iterrows():
        nid = str(r["node_id"])
        lab = str(r.get("label", "")).strip()
        if not lab or len(lab) < 2:
            continue
        ntype = str(r.get("node_type", ""))
        lab_l = lab.lower()
        defn = str(r.get("definition", "")).lower() if pd.notna(r.get("definition")) else ""
        hay = f"{lab_l} {defn}"
        token_hay = lab_l if ntype in ("Disease", "Procedure", "DiagnosticTest", "Drug", "BodySystem") else hay

        if lab_l in low and len(lab_l) >= 3:
            if ntype == "Disease" and not is_valid_graph_label(lab):
                continue
            add_id(nid, f'label substring in query ("{lab[:40]}")')
            continue

        if any(
            re.search(rf"(?<!\w){re.escape(t)}(?!\w)", token_hay) for t in tokens if len(t) >= 4
        ):
            if ntype == "Disease" and not is_valid_graph_label(lab):
                continue
            add_id(nid, f'token<->node label{" (+def)" if token_hay is hay else ""} ("{lab[:40]}")')
            continue

        lab_words = set(re.findall(r"[a-z0-9]+", lab_l))
        if lab_words & set(tokens):
            if ntype == "Disease" and not is_valid_graph_label(lab):
                pass
            else:
                add_id(nid, f'word overlap with node label ("{lab[:40]}")')

        if ntype == "Symptom":
            for t in tokens:
                if len(t) >= 4 and t in lab_l:
                    add_id(nid, f'token "{t}" in Symptom label "{lab[:40]}"')
                    break

    if tokens and RESPIRATORY_EXPERIENCE_TOKENS & set(tokens):
        for _, r in nodes[nodes["node_type"] == "Symptom"].iterrows():
            lab_s = str(r.get("label", "")).strip()
            lab_l = lab_s.lower()
            if "breath" in lab_l or "wheez" in lab_l:
                add_id(
                    str(r["node_id"]),
                    f'respiratory wording in query -> Symptom "{lab_s[:45]}"',
                )

    if symptom_vocab:
        for canonical, aliases in symptom_vocab.items():
            hit = False
            cas_l = canonical.lower()
            if cas_l in low and len(cas_l) > 3:
                hit = True
            if not hit:
                for a in aliases:
                    if len(a) >= 3 and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", low):
                        hit = True
                        break
            if not hit:
                continue
            title = " ".join(w.capitalize() for w in cas_l.split())
            for _, r in nodes[nodes["node_type"] == "Symptom"].iterrows():
                sl = str(r.get("label", "")).strip().lower()
                if sl == cas_l or sl.replace(" ", "") == cas_l.replace(" ", ""):
                    add_id(str(r["node_id"]), f'symptom vocabulary "{canonical}"')
                    break
            else:
                for _, r in nodes[nodes["node_type"] == "Symptom"].iterrows():
                    sl = str(r.get("label", "")).strip().lower()
                    if title.lower() == sl or canonical.lower() in sl:
                        add_id(str(r["node_id"]), f'symptom vocabulary "{canonical}"->graph "{sl}"')
                        break

    return seeds, reasons


def refine_glossary_seeds(
    user_text: str,
    glossary: pd.DataFrame,
    label_index: DefaultDict[str, List[str]],
    nodes: pd.DataFrame,
    seeds: Set[str],
    reasons: List[str],
    skip_term_if_topic: Optional[str] = None,
) -> None:
    low = norm_user(user_text)
    topic_l = norm_user(skip_term_if_topic) if skip_term_if_topic else ""
    for _, g in glossary.iterrows():
        term = str(g.get("term", "")).strip()
        if len(term) < 3:
            continue
        if topic_l and norm_user(term) == topic_l:
            continue
        if not re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", low):
            continue
        assoc = str(g.get("associated_entries", "") or "")
        for p in re.split(r"[;]", assoc):
            p = clean_text(p)
            if len(p) < 3 or not is_valid_entry_name(p):
                continue
            for nid in label_to_node_ids(p, label_index, nodes):
                if nid not in seeds:
                    seeds.add(nid)
                    reasons.append(f'glossary "{term}" -> entry "{p[:50]}" -> {nid}')


def bfs_context(
    seeds: Set[str],
    adj: DefaultDict[str, List[Tuple[str, str, str, str]]],
    by_id: Dict[str, dict],
    max_nodes: int = MAX_BFS_NODES,
) -> Tuple[Dict[str, int], Set[str]]:
    """Multi-source BFS; return distance map and visited set."""
    dist: Dict[str, int] = {}
    q: deque[str] = deque()
    for s in seeds:
        if s in by_id:
            dist[s] = 0
            q.append(s)
    visited: Set[str] = set(dist.keys())
    while q:
        if len(visited) >= max_nodes:
            break
        u = q.popleft()
        du = dist[u]
        for v, _, _, _ in adj.get(u, []):
            if v not in by_id:
                continue
            if v not in visited:
                visited.add(v)
                dist[v] = du + 1
                if len(visited) < max_nodes:
                    q.append(v)
    return dist, visited


def normalize_neo4j_uri_for_local(uri: str) -> str:
    """neo4j:// on single-instance Desktop often fails routing; use bolt:// for localhost."""
    u = (uri or "").strip()
    if not u:
        return "bolt://localhost:7687"
    p = urlparse(u)
    if p.scheme != "neo4j":
        return u
    h = (p.hostname or "").lower()
    if h in ("localhost", "127.0.0.1", "::1", ""):
        port = p.port or 7687
        return f"bolt://{h or '127.0.0.1'}:{port}"
    return u


def bfs_context_neo4j(
    seeds: Set[str],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    by_id: Dict[str, dict],
    max_nodes: int = MAX_BFS_NODES,
    max_depth: int = 4,
) -> Tuple[Dict[str, int], Set[str]]:
    """Neo4j traversal backend (optional)."""
    from neo4j import GraphDatabase

    if not seeds:
        return {}, set()

    seed_list = [s for s in seeds if s in by_id]
    if not seed_list:
        return {}, set()

    q = f"""
    UNWIND $seed_ids AS sid
    MATCH (s:MedicalNode {{node_id: sid}})
    MATCH p=(s)-[:AFFECTS|HAS_SYMPTOM|TREATS*0..{int(max_depth)}]-(v:MedicalNode)
    WITH v.node_id AS node_id, min(length(p)) AS dist
    ORDER BY dist ASC, node_id ASC
    LIMIT $limit
    RETURN node_id, dist
    """
    dist: Dict[str, int] = {}
    visited: Set[str] = set()
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session(database=neo4j_database) as session:
            rows = session.run(q, seed_ids=seed_list, limit=max_nodes)
            for r in rows:
                nid = str(r["node_id"])
                if nid not in by_id:
                    continue
                d = int(r["dist"])
                if nid not in dist or d < dist[nid]:
                    dist[nid] = d
                visited.add(nid)
    finally:
        driver.close()
    return dist, visited


def disease_scores_from_graph(
    dist: Dict[str, int],
    visited: Set[str],
    by_id: Dict[str, dict],
    seeds: Set[str],
    adj: DefaultDict[str, List[Tuple[str, str, str, str]]],
) -> List[Tuple[float, str]]:
    """Score Disease nodes using distance + direct seed incidence on incident edges."""
    disease_nid_score: DefaultDict[str, float] = defaultdict(float)
    for nid in visited:
        info = by_id.get(nid)
        if not info or info["node_type"] != "Disease":
            continue
        d = dist.get(nid, 99)
        base = 100.0 / (1.0 + d)
        disease_nid_score[nid] += base
        for nb, rel, _, _ in adj.get(nid, []):
            if nb in seeds:
                if rel == "HAS_SYMPTOM":
                    disease_nid_score[nid] += 25.0
                elif rel == "AFFECTS":
                    disease_nid_score[nid] += 12.0
                elif rel == "TREATS":
                    disease_nid_score[nid] += 2.0

    ranked = [
        (sc, by_id[nid]["label"])
        for nid, sc in disease_nid_score.items()
        if is_valid_graph_label(by_id[nid]["label"])
    ]
    ranked.sort(key=lambda x: -x[0])
    return ranked


def format_edges_for_disease(
    disease_label: str,
    edges: pd.DataFrame,
    limit: int = EDGE_LINES_PER_DISEASE,
) -> str:
    sub = edges[
        (edges["source_label"].astype(str) == disease_label)
        | (edges["target_label"].astype(str) == disease_label)
    ]
    lines: List[str] = []
    for _, e in sub.head(limit).iterrows():
        lines.append(
            f"{e['source_label']} --{e['relationship']}--> {e['target_label']} (evidence={e.get('evidence', '')})"
        )
    if not lines:
        return "(no edges listing this label in edge table)"
    return "\n".join(lines)


def matrix_row_blurb(matrix: pd.DataFrame, disease_name: str) -> str:
    col = "disease_name" if "disease_name" in matrix.columns else "entry_name"
    if disease_name not in set(matrix[col].astype(str)):
        return ""
    row = matrix[matrix[col].astype(str) == disease_name].iloc[0]
    sym_cols = [
        c
        for c in matrix.columns
        if c.lower() not in {"disease", "disease_name", "entry_name", "entry_type"}
    ]
    pos = [c for c in sym_cols if str(row.get(c, 0)) in {"1", "1.0", "True", "true"}]
    if not pos:
        return "matrix: no positive symptom flags in binary matrix for this disease."
    return "matrix positives (1): " + ", ".join(pos[:40]) + (" ..." if len(pos) > 40 else "")


def build_evidence_prompt(
    user_text: str,
    seed_reasons: List[str],
    ranked_diseases: List[Tuple[float, str]],
    entries: pd.DataFrame,
    matrix: pd.DataFrame,
    edges: pd.DataFrame,
    visited_summary: str,
) -> str:
    dis_blocks: List[str] = []
    for score, dlabel in ranked_diseases[:TOP_CANDIDATE_EXCERPTS]:
        sub = entries[entries["entry_name"].astype(str) == dlabel]
        enc = (
            format_encyclopedia_context(sub.iloc[0])
            if len(sub)
            else "(no encyclopedia row for this label)"
        )
        eblk = format_edges_for_disease(dlabel, edges)
        mblk = matrix_row_blurb(matrix, dlabel)
        dis_blocks.append(
            f"### Disease: {dlabel} (graph score {score:.1f})\n**Local graph edges:**\n{eblk}\n**Encyclopedia:**\n{enc}\n**{mblk}**\n"
        )
    body = "\n".join(dis_blocks)
    seed_txt = "\n".join(seed_reasons[:40]) if seed_reasons else "(no direct node label hits - BFS may still reach diseases)"

    return f"""
You are a medical **educational** assistant. Your ONLY evidence is the material below from a structured
dump (knowledge graph + encyclopedia CSV + optional binary symptom matrix row). Do **not** invent facts
that are not supported by these excerpts.

**User message:** {user_text}

**How retrieval worked (for transparency):**
- Seed graph nodes were matched from: node labels/definitions vs user text, glossary terms -> linked entry names,
  and vocabulary aligned to the dataset's symptom columns (symptom_aliases.json).
- The graph was traversed (undirected BFS, capped) from those seeds. Disease nodes are ranked by proximity
  and by direct HAS_SYMPTOM / AFFECTS / weak TREATS links to seeds.
- TREATS edges are known to be noisy in this export; treat them as low confidence.

**Seeds / match log (first lines):**
{seed_txt}

**Visited region summary:** {visited_summary}

**Ranked diseases + encyclopedia excerpts + local edges + matrix note:**
{body}

**Your tasks:**
1. Infer what the user is asking (symptom triage vs definition vs mixed) using only the evidence bundle.
2. Explain **which diseases** in the bundle are most consistent, citing **short quotes or fields** you saw.
3. If the bundle is thin, say so and list what is missing.
4. Red flags & "see a clinician" where appropriate.
5. Clear disclaimer: not a diagnosis.

Output format rules (important):
- Use plain text only.
- Do not use markdown symbols like **, *, #, -, |, or ---.
- Do not use tables.
- Keep it concise and readable for a basic UI text box.
- Use numbered lines like "1) ...", "2) ...".
"""


def build_definition_prompt(user_text: str, row: pd.Series, edges: pd.DataFrame, nodes: pd.DataFrame) -> str:
    name = str(row["entry_name"])
    enc = format_encyclopedia_context(row)
    eblk = format_edges_for_disease(name, edges, limit=40)
    nid_rows = nodes[nodes["label"].astype(str) == name]
    nstub = ""
    if len(nid_rows):
        r = nid_rows.iloc[0]
        nstub = f"node_id={r.get('node_id')} type={r.get('node_type')}"
    return f"""
User question: {user_text}

**Matched encyclopedia entry:** {name}

**Graph edges touching this label:**
{eblk}

**Encyclopedia fields:**
{enc}

**Node stub:** {nstub or "none"}

Answer using only the above. If something is absent, say it is not in the dataset.
Format rules:
- Plain text only (no markdown symbols, no tables).
- Use short numbered lines (1), 2), 3)...).
- End with one-line disclaimer: "This is educational information, not a diagnosis."
"""


def query_openrouter(api_key: str, model: str, prompt: str, base_url: Optional[str] = None) -> str:
    """Call OpenRouter chat completions endpoint (OpenAI-compatible)."""
    import httpx

    base = (base_url or os.getenv("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL_DEFAULT).rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_X_TITLE", "MediBot")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt.strip()}],
        "temperature": 0.25,
        "max_tokens": 4096,
    }
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as ex:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from ex


def call_llm(prompt: str, model: str, api_key: str) -> str:
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY")
    return query_openrouter(api_key, model, prompt)


def default_model_for() -> str:
    return DEFAULT_OPENROUTER_MODEL


def configure_console_output() -> None:
    """Avoid Windows cp1252 crashes when model text includes Unicode punctuation."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> None:
    configure_console_output()
    load_dotenv()
    ap = argparse.ArgumentParser(description="KG-first MediBot (OpenRouter GPT OSS 120B)")
    ap.add_argument("--dataset-dir", default="final_dataset")
    ap.add_argument("--user-input", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--backend", choices=["csv", "neo4j"], default="csv")
    ap.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    ap.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    ap.add_argument("--neo4j-max-depth", type=int, default=4)
    args = ap.parse_args()

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not args.no_llm and not openrouter_key:
        raise ValueError("Set OPENROUTER_API_KEY")

    model = args.model or default_model_for()
    ddir = Path(args.dataset_dir)

    entries = pd.read_csv(ddir / "01_medical_encyclopedia_entries.csv")
    matrix = pd.read_csv(ddir / "04_disease_symptom_matrix.csv")
    nodes = pd.read_csv(ddir / "02_knowledge_graph_nodes.csv")
    edges = pd.read_csv(ddir / "03_knowledge_graph_edges.csv")
    glossary = pd.read_csv(ddir / "05_medical_glossary.csv")
    symptom_vocab = load_symptom_alias_vocabulary(ddir)

    by_id, label_index, adj = build_graph_indices(nodes, edges)
    user_in = args.user_input

    definition_topic = parse_definition_query(user_in)
    if definition_topic:
        row = find_encyclopedia_row(entries, definition_topic)
        print("\n=== Intent: definition / topic lookup ===")
        print(definition_topic)
        if row is None and not args.no_llm:
            prompt = (
                f"User: {user_in}\nNo matching encyclopedia row. Say so briefly. Not medical advice."
            )
        elif row is None:
            prompt = ""
        else:
            print("Matched:", row["entry_name"])
            if not args.no_llm:
                prompt = build_definition_prompt(user_in, row, edges, nodes)
            else:
                prompt = ""
            print("\n--- Encyclopedia (preview) ---\n")
            print(format_encyclopedia_context(row)[:2800])
        print("\n=== Seed graph nodes ===")
        seeds, seed_reasons = match_seed_nodes(user_in, nodes, symptom_vocab)
        refine_glossary_seeds(
            user_in, glossary, label_index, nodes, seeds, seed_reasons, skip_term_if_topic=definition_topic
        )
        if row is not None:
            for nid in label_to_node_ids(str(row["entry_name"]), label_index, nodes):
                if nid not in seeds:
                    seeds.add(nid)
                    seed_reasons.append(f"definition topic node -> {nid}")
        for line in seed_reasons[:25]:
            print(line)
        if args.no_llm:
            return
        print("\n=== OpenRouter response ===\n")
        print(call_llm(prompt, model, openrouter_key or ""))
        return

    seeds, seed_reasons = match_seed_nodes(user_in, nodes, symptom_vocab)
    refine_glossary_seeds(user_in, glossary, label_index, nodes, seeds, seed_reasons)

    print("\n=== Graph seeds (node_ids) ===")
    print(", ".join(sorted(seeds)) if seeds else "(none - query may still match after expansion)")

    if args.backend == "neo4j":
        if not args.neo4j_password:
            raise ValueError("Set NEO4J_PASSWORD (or pass --neo4j-password) when --backend neo4j")
        neo_uri = normalize_neo4j_uri_for_local(args.neo4j_uri)
        if neo_uri != args.neo4j_uri:
            print(f"\nNote: Neo4j URI adjusted for local desktop: {neo_uri}")
        try:
            dist, visited = bfs_context_neo4j(
                seeds=seeds,
                neo4j_uri=neo_uri,
                neo4j_user=args.neo4j_user,
                neo4j_password=args.neo4j_password,
                neo4j_database=args.neo4j_database,
                by_id=by_id,
                max_nodes=MAX_BFS_NODES,
                max_depth=args.neo4j_max_depth,
            )
            print(f"\n=== Graph backend ===\nneo4j ({neo_uri}, db={args.neo4j_database})")
        except Exception as ex:
            print(f"\n[warn] neo4j backend failed ({ex}); falling back to csv traversal.")
            dist, visited = bfs_context(seeds, adj, by_id)
            print("\n=== Graph backend ===\ncsv (fallback)")
    else:
        dist, visited = bfs_context(seeds, adj, by_id)
        print("\n=== Graph backend ===\ncsv")
    ranked = disease_scores_from_graph(dist, visited, by_id, seeds, adj)

    if not ranked and seeds:
        for nid in list(seeds)[:5]:
            for nb, _, _, _ in adj.get(nid, []):
                if by_id.get(nb, {}).get("node_type") == "Disease":
                    lb = by_id[nb]["label"]
                    if is_valid_graph_label(lb):
                        ranked.append((50.0, lb))
        ranked.sort(key=lambda x: -x[0])

    if not ranked:
        tokens = user_word_tokens(user_in)
        for _, r in entries[
            entries["entry_type"].astype(str).str.contains("Disease", na=False, case=False)
        ].iterrows():
            if not is_valid_entry_name(str(r["entry_name"])):
                continue
            blob = norm_user(" ".join(str(r.get(c, "")) for c in ENCYCLOPEDIA_TEXT_COLUMNS if c in r.index))
            sc = sum(1 for t in tokens if len(t) > 3 and t in blob)
            if sc >= 2:
                ranked.append((float(sc), str(r["entry_name"])))
        ranked.sort(key=lambda x: -x[0])
        ranked = ranked[:TOP_DISEASES]

    ranked = ranked[:TOP_DISEASES]

    visited_types: DefaultDict[str, int] = defaultdict(int)
    for vid in visited:
        t = by_id.get(vid, {}).get("node_type", "?")
        visited_types[t] += 1
    visited_summary = ", ".join(f"{k}={v}" for k, v in sorted(visited_types.items()))

    print("\n=== Ranked diseases (graph-traversal score) ===")
    for sc, lab in ranked[:15]:
        print(f"{lab[:70]:<70} {sc:.1f}")

    print("\n=== Seed trace (sample) ===")
    for line in seed_reasons[:25]:
        print(line)

    if args.no_llm:
        print("\n(--no-llm: skipping model)")
        return

    prompt = build_evidence_prompt(
        user_in,
        seed_reasons,
        ranked,
        entries,
        matrix,
        edges,
        visited_summary,
    )

    print("\n=== OpenRouter response ===\n")
    print(call_llm(prompt, model, openrouter_key or ""))


if __name__ == "__main__":
    main()






