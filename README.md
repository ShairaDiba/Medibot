# 🏥 MediBot — Knowledge Graph-First Medical Assistant

MediBot is an AI-powered medical assistant that retrieves evidence from a structured medical knowledge graph before generating answers via an LLM. It is designed to provide grounded, evidence-based responses — not hallucinations.

> ⚠️ **Disclaimer:** MediBot is an educational prototype. It is not medical advice, diagnosis, or treatment.

---

## ✨ Features

- 🔍 **Knowledge Graph Retrieval** — BFS traversal over medical nodes and edges
- 📖 **Encyclopedia Lookup** — Disease/topic definitions from a structured dataset
- 🧬 **Disease × Symptom Matching** — Ranked disease suggestions based on symptoms
- 🤖 **LLM Answering via OpenRouter** — Answers grounded strictly in retrieved evidence
- 🌐 **Browser UI** — Simple one-page frontend with query input and results display
- 🇧🇩 **Bangla Translation** — One-click translation of answers to Bangla

---

## 👥 Team

| Name | GitHub |
|------|--------|
| Shruti Khisa | [@ShrutiKhisa](https://github.com/ShrutiKhisa) |
| Farhan Tanvir | [@FarhanTanvir](https://github.com/FarhanTanvir) |
| Shaira Akhter Diba | [@ShairaDiba](https://github.com/ShairaDiba) |
| Fazli Rabbi Noor | [@FarhanNoor](https://github.com/FarhanNoor) |

---

## 🚀 Quickstart

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Create a `.env` file** in the project root:

```env
# Required
OPENROUTER_API_KEY=your_openrouter_key

# Optional OpenRouter settings
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=https://your-app.example
OPENROUTER_X_TITLE=MediBot

# Optional — only needed for Neo4j backend
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

**Run via CLI:**

```bash
python medibot.py --user-input "I have trouble breathing and chest tightness"
```

**Run the browser UI:**

```bash
python ui_server.py
```

Then open: `http://127.0.0.1:8787/UI.html`

---

## 🗂️ Project Structure

```
medibot.py                  # Main pipeline
ui_server.py                # Local server for the browser UI
UI.html                     # Single-file frontend
load_kg_to_neo4j.py         # Imports graph into Neo4j
final_dataset/
├── 01_medical_encyclopedia_entries.csv
├── 02_knowledge_graph_nodes.csv
├── 03_knowledge_graph_edges.csv
├── 04_disease_symptom_matrix.csv
├── 05_medical_glossary.csv
└── symptom_aliases.json
```

---

## ⚙️ How It Works

### 1. Query Classification
MediBot first checks whether the query is a **definition request** (e.g., *"what is asthma"*) or a **symptom triage** (e.g., *"I have fever and cough"*), and routes accordingly.

### 2. Seed Node Matching
User input is mapped to knowledge graph node IDs via:
- Direct label matching
- Token-to-label matching
- Symptom alias expansion (`symptom_aliases.json`)
- Glossary bridging

### 3. Graph Traversal (BFS)
Starting from seed nodes, MediBot runs BFS across the knowledge graph using either:
- **CSV backend** (default) — in-memory traversal
- **Neo4j backend** (optional) — scalable Cypher-based traversal

### 4. Disease Ranking
Visited disease nodes are scored by graph proximity and edge type bonuses (`HAS_SYMPTOM`, `AFFECTS`, `TREATS`).

### 5. Evidence Assembly
For top-ranked diseases, MediBot collects encyclopedia entries, graph edges, and symptom matrix data into a bounded evidence bundle.

### 6. LLM Response
The evidence bundle is sent to OpenRouter (default model: `openai/gpt-oss-120b`) with a strict instruction to answer **only from the provided evidence**.

---

## 🖥️ CLI Options

```bash
# Basic query
python medibot.py --user-input "I have fever and cough"

# Change model
python medibot.py --model openai/gpt-oss-120b --user-input "I have fever and cough"

# Retrieval only (no LLM call)
python medibot.py --no-llm --user-input "I have fever and cough"

# Use Neo4j backend
python medibot.py --backend neo4j --user-input "I have fever and cough"
```

---

## 🔗 Neo4j Setup (Optional)

Load the graph into Neo4j for visualization:

```bash
python load_kg_to_neo4j.py --dataset-dir final_dataset --clear
```

Example Cypher queries:

```cypher
MATCH (n:MedicalNode {label: "Asthma"}) RETURN n LIMIT 1;
```

```cypher
MATCH (n:MedicalNode {node_id: "D00108"})-[:HAS_SYMPTOM|AFFECTS|TREATS*1..3]-(m)
RETURN DISTINCT m.label LIMIT 25;
```

> Use `bolt://127.0.0.1:7687` for local Neo4j Desktop instances.

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `UnicodeEncodeError` on Windows | Use Python 3.10+, restart terminal |
| `OpenRouter API key missing` | Add key to `.env` or paste into UI field |
| Neo4j routing error with `neo4j://...` | Use `bolt://127.0.0.1:7687` instead |
