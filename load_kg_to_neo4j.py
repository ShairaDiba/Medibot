#!/usr/bin/env python3
"""
Load final_dataset knowledge graph CSVs into Neo4j.

Prereqs:
  pip install neo4j pandas python-dotenv

Neo4j options:
  - Neo4j Desktop: start a local DB, set password, note bolt URI (bolt://localhost:7687).
  - Docker: docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/yourpass neo4j:latest
  - Aura: use the neo4j+s:// URI from the console.

Env (optional, override CLI):
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

Usage:
  python load_kg_to_neo4j.py --dataset-dir final_dataset
  python load_kg_to_neo4j.py --dataset-dir final_dataset --clear

Model:
  (:MedicalNode {node_id, node_type, label, ...}) with real relationship types
  :AFFECTS, :HAS_SYMPTOM, :TREATS and properties {edge_id, weight, evidence}.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from urllib.parse import urlparse

NODE_BATCH = 1500
EDGE_BATCH = 2000

ALLOWED_RELATIONSHIPS = frozenset({"AFFECTS", "HAS_SYMPTOM", "TREATS"})


def normalize_uri_for_desktop(uri: str) -> str:
    """
    Neo4j Desktop runs a single instance: neo4j:// often triggers 'Unable to retrieve routing information'.
    Use bolt:// for localhost / 127.0.0.1 when the scheme is neo4j.
    """
    uri = (uri or "").strip()
    if not uri:
        return "bolt://localhost:7687"
    p = urlparse(uri)
    if p.scheme != "neo4j":
        return uri
    host = (p.hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1", ""):
        port = p.port or 7687
        return f"bolt://{host or '127.0.0.1'}:{port}"
    return uri


def _row_to_props(row: pd.Series) -> Dict[str, Any]:
    raw = row.to_dict()
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if pd.isna(v):
            continue
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int,)):
            out[k] = int(v)
        elif isinstance(v, float):
            out[k] = float(v)
        else:
            s = str(v).strip()
            if s:
                out[k] = s
    out["node_id"] = str(row["node_id"])
    return out


def clear_db(driver: Any, database: str) -> None:
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")


def ensure_schema(driver: Any, database: str) -> None:
    with driver.session(database=database) as session:
        session.run(
            """
            CREATE CONSTRAINT medical_node_id_unique IF NOT EXISTS
            FOR (n:MedicalNode) REQUIRE n.node_id IS UNIQUE
            """
        )


def load_nodes(driver: Any, database: str, nodes_csv: Path) -> int:
    df = pd.read_csv(nodes_csv)
    count = 0
    props_list = [_row_to_props(df.iloc[i]) for i in range(len(df))]
    # Neo4j properties: drop keys neo4j can't store — keep JSON-serializable scalars only
    for i in range(0, len(props_list), NODE_BATCH):
        batch = props_list[i : i + NODE_BATCH]
        with driver.session(database=database) as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:MedicalNode {node_id: row.node_id})
                SET n += row
                """,
                rows=batch,
            )
        count += len(batch)
    return count


def load_edges(driver: Any, database: str, edges_csv: Path) -> int:
    df = pd.read_csv(edges_csv)
    df = df[df["relationship"].isin(ALLOWED_RELATIONSHIPS)]
    total = 0
    for rel_type, chunk in df.groupby("relationship", sort=False):
        rel = str(rel_type)
        sub = chunk.reset_index(drop=True)
        for i in range(0, len(sub), EDGE_BATCH):
            part = sub.iloc[i : i + EDGE_BATCH]
            rows: List[Dict[str, Any]] = []
            for _, r in part.iterrows():
                rows.append(
                    {
                        "source_node_id": str(r["source_node_id"]),
                        "target_node_id": str(r["target_node_id"]),
                        "edge_id": str(r.get("edge_id", "")),
                        "weight": float(r["weight"]) if pd.notna(r.get("weight")) else None,
                        "evidence": str(r.get("evidence", "")) if pd.notna(r.get("evidence")) else None,
                    }
                )
            cypher = f"""
            UNWIND $rows AS row
            MATCH (a:MedicalNode {{node_id: row.source_node_id}})
            MATCH (b:MedicalNode {{node_id: row.target_node_id}})
            MERGE (a)-[r:`{rel}`]->(b)
            SET r.edge_id = row.edge_id,
                r.weight = row.weight,
                r.evidence = row.evidence
            """
            with driver.session(database=database) as session:
                session.run(cypher, rows=rows)
            total += len(rows)
    return total


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Load knowledge graph CSVs into Neo4j")
    ap.add_argument("--dataset-dir", default="final_dataset")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    ap.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    ap.add_argument("--clear", action="store_true", help="Delete all nodes/relationships first")
    args = ap.parse_args()

    if not args.password:
        raise SystemExit("Set NEO4J_PASSWORD or pass --password")

    args.uri = normalize_uri_for_desktop(args.uri)
    if args.uri.startswith("bolt://"):
        print(f"Note: using direct Bolt URI for local Neo4j: {args.uri}")

    ddir = Path(args.dataset_dir)
    nodes_f = ddir / "02_knowledge_graph_nodes.csv"
    edges_f = ddir / "03_knowledge_graph_edges.csv"
    if not nodes_f.is_file() or not edges_f.is_file():
        raise SystemExit(f"Missing {nodes_f} or {edges_f}")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        driver.verify_connectivity()
        if args.clear:
            clear_db(driver, args.database)
        ensure_schema(driver, args.database)
        n = load_nodes(driver, args.database, nodes_f)
        e = load_edges(driver, args.database, edges_f)
        print(f"Loaded {n} MedicalNode rows, {e} relationships (batched MERGE).")
        print("Try in Neo4j Browser:\n")
        print('  MATCH (n:MedicalNode {label: "Asthma"}) RETURN n LIMIT 1;')
        print(
            "  MATCH (n:MedicalNode {node_id: 'D00108'})-[:HAS_SYMPTOM|AFFECTS|TREATS*1..3]-(m) RETURN DISTINCT m.label LIMIT 25;"
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
