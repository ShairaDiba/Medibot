# MediBot (KG-first Medical Assistant)

MediBot is a **knowledge-graph-first** medical assistant built on a local dataset export (`final_dataset/`).
It retrieves relevant evidence from:

- a medical **knowledge graph** (nodes + edges),
- an **encyclopedia** dataset (disease/topic entries),
- a **disease × symptom matrix**,
- a **glossary** (terms -> related entries),

…and then asks an LLM (via **OpenRouter**) to answer **using only the retrieved evidence**.

It supports two graph traversal backends:

- **CSV backend** (default): in-memory BFS over `03_knowledge_graph_edges.csv`
- **Neo4j backend** (optional): BFS-style traversal in Neo4j for scalability/visualization

This repo also includes a simple browser UI (`UI.html`) connected to the Python pipeline via a local server (`ui_server.py`),
with a one-click **Bangla translation** feature.

## Quickstart

Install deps:

```bash
pip install -r requirements.txt
```

Create `.env` in the project root (same folder as `medibot.py`):

```env
# Required for LLM answers and Bangla translation
OPENROUTER_API_KEY=your_openrouter_key

# Optional OpenRouter settings
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=https://your-app.example
OPENROUTER_X_TITLE=MediBot

# Optional (only if using --backend neo4j)
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

Run the CLI:

```bash
python medibot.py --user-input "I have trouble breathing and chest tightness"
```

Run the UI:

```bash
python ui_server.py
```

Then open `http://127.0.0.1:8787/UI.html`.

## Project layout

- `medibot.py`
  - Main pipeline: loads datasets, matches seeds, traverses graph, ranks diseases, builds evidence prompt, calls OpenRouter.
- `ui_server.py`
  - Local server for the UI.
  - `POST /api/query`: executes `medibot.py` and returns structured JSON for the UI.
  - `POST /api/translate`: translates the current answer to Bangla using OpenRouter.
- `UI.html`
  - Single-file frontend.
  - Calls the local server endpoints and displays: seeds, ranked diseases, traces, and the final answer.
  - Buttons:
    - **Translate to Bangla**
    - **Show Original English**
- `load_kg_to_neo4j.py`
  - Imports graph nodes/edges CSVs into Neo4j as `(:MedicalNode)` and relationships.
- `final_dataset/`
  - `01_medical_encyclopedia_entries.csv`
  - `02_knowledge_graph_nodes.csv`
  - `03_knowledge_graph_edges.csv`
  - `04_disease_symptom_matrix.csv`
  - `05_medical_glossary.csv`
  - `symptom_aliases.json`

## How MediBot works (end-to-end)

### 1) Input classification: definition vs symptom triage

`medibot.py` first checks whether the user query looks like a definition request:

- examples: `what is asthma`, `define pneumonia`, `tell me about migraine`

If it is a definition query:

- it finds the best matching row in `01_medical_encyclopedia_entries.csv`
- prints an encyclopedia preview
- builds a **definition prompt** containing:
  - encyclopedia fields
  - graph edges touching that label
  - node stub (`node_id`, `node_type`) if present

If it is not a definition query:

- it runs the **graph-first retrieval** flow described below.

### 2) Seed node matching (turn user text into graph node_ids)

Seeds are node IDs in the knowledge graph that match the user message. Seeds come from multiple sources:

- **Direct label hits**: node `label` text appears in the user message
- **Token-to-label matches**: user tokens match node label/definition
- **Symptom vocabulary mapping**: `final_dataset/symptom_aliases.json` expands symptom phrases into dataset-native symptom names
- **Glossary bridging**: `05_medical_glossary.csv` maps matched terms to associated encyclopedia entries, which then map back to KG nodes
- **Small “experience” bridge**: common respiratory wording maps toward breathing-related symptom nodes

The pipeline keeps a “seed reason log” so you can see why each seed was included.

### 3) Graph traversal (BFS) from seeds

MediBot then traverses the knowledge graph starting at the seed node IDs.

Two backends are supported:

- **CSV backend** (default): builds an undirected adjacency list from `03_knowledge_graph_edges.csv` and runs multi-source BFS.
- **Neo4j backend**: runs a Cypher query to retrieve reachable nodes up to a max hop count.

Traversal is capped (e.g., `MAX_BFS_NODES`) so prompts stay bounded and the UI stays responsive.

### 4) Disease ranking (graph proximity + edge bonuses)

From the visited region, MediBot scores disease nodes:

- base score comes from proximity (distance in BFS)
- bonus score if a disease has direct edges to seed nodes:
  - `HAS_SYMPTOM`: strong bonus
  - `AFFECTS`: medium bonus
  - `TREATS`: weak bonus (this dataset contains noisy TREATS edges; treated as low confidence)

The output is a list of top diseases with graph scores.

### 5) Evidence bundle assembly (what the model is allowed to use)

For the top ranked diseases, MediBot collects:

- **Encyclopedia fields** from `01_medical_encyclopedia_entries.csv`
- **Local graph edge lines** touching the disease label from `03_knowledge_graph_edges.csv`
- **Symptom matrix positives** from `04_disease_symptom_matrix.csv` (if present for that disease)

This evidence bundle is what gets fed into the LLM prompt.

### 6) LLM response (OpenRouter)

The LLM is called via OpenRouter’s OpenAI-compatible endpoint:

- default model: `openai/gpt-oss-120b`
- the prompt explicitly instructs:
  - “use only the provided evidence”
  - “if missing, say it is missing”
  - “end with a disclaimer”
  - “plain text output” (no markdown tables/symbols)

## Using the CLI (`medibot.py`)

Basic usage:

```bash
python medibot.py --user-input "I have fever and cough"
```

Change the model:

```bash
python medibot.py --model openai/gpt-oss-120b --user-input "I have fever and cough"
```

No-LLM mode (retrieval only; useful for debugging / no API key):

```bash
python medibot.py --no-llm --user-input "I have fever and cough"
```

Neo4j traversal backend:

```bash
python medibot.py --backend neo4j --user-input "I have fever and cough"
```

## Using the UI (`UI.html` + `ui_server.py`)

Start the server:

```bash
python ui_server.py
```

Open: `http://127.0.0.1:8787/UI.html`

The UI:

- sends your query to `POST /api/query`
- shows seeds, ranking, and a simplified trace
- displays the final answer returned by `medibot.py`

Bangla translation:

- click **Translate to Bangla**
- the UI calls `POST /api/translate` with the current answer text
- server uses the OpenRouter key from:
  - the UI “API key” field, or
  - `.env` (`OPENROUTER_API_KEY`)
- click **Show Original English** to toggle back without rerunning the pipeline

## Neo4j: load the graph for visualization and traversal

Load nodes/edges into Neo4j:

```bash
python load_kg_to_neo4j.py --dataset-dir final_dataset --clear
```

Notes:

- Neo4j Desktop local instances often require `bolt://...` (not `neo4j://...`).
- `load_kg_to_neo4j.py` auto-normalizes `neo4j://localhost` -> `bolt://localhost`.

Example Cypher in Neo4j Browser:

```cypher
MATCH (n:MedicalNode {label: "Asthma"}) RETURN n LIMIT 1;
```

```cypher
MATCH (n:MedicalNode {node_id: "D00108"})-[:HAS_SYMPTOM|AFFECTS|TREATS*1..3]-(m)
RETURN DISTINCT m.label LIMIT 25;
```

## Troubleshooting

### “UnicodeEncodeError / UnicodeDecodeError” on Windows

This project forces UTF-8 output in `medibot.py` and forces UTF-8 decode in `ui_server.py`.
If you still see encoding issues, restart the terminal and ensure you are using Python 3.10+.

### “OpenRouter API key missing”

- Put `OPENROUTER_API_KEY=...` in `.env` (project root), and restart `ui_server.py`, or
- Paste the key into the UI “OpenRouter API key” field.

### Neo4j routing errors with `neo4j://...`

Use `bolt://127.0.0.1:7687` for local Desktop instances (single instance, no routing).

## Disclaimer

MediBot is an educational prototype. It is not medical advice, diagnosis, or treatment.
