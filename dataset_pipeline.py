"""
=============================================================================
Medical Knowledge Graph Dataset Pipeline
=============================================================================
Source   : The Gale Encyclopedia of Medicine, 5th Edition (2015)
           Jacqueline L. Longe (Ed.), Gale/Cengage Learning, Vols. 1-9
Author   : [Your Name]
Project  : Knowledge-Graph Augmented NLP Framework for Medical Diseases

Description:
    This pipeline parses the Gale Encyclopedia of Medicine PDF, extracts
    structured medical knowledge, and produces 5 datasets for use in a
    knowledge-graph augmented NLP framework:

    Dataset 1 → 01_medical_encyclopedia_entries.csv   (1,803 entries)
    Dataset 2 → 02_knowledge_graph_nodes.csv          (1,853 nodes)
    Dataset 3 → 03_knowledge_graph_edges.csv          (50,000+ edges)
    Dataset 4 → 04_disease_symptom_matrix.csv         (1,464 × 65 matrix)
    Dataset 5 → 05_medical_glossary.csv               (2,464 terms)

Dependencies:
    pip install pdfplumber pypdf pandas tqdm
    sudo apt install poppler-utils   # for pdftotext

Usage:
    python pipeline.py --pdf "Gale_Encyclopedia_of_Medicine_5th_Ed.pdf"
=============================================================================
"""

import subprocess
import re
import csv
import os
import json
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm  # pip install tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

PDF_PATH        = "Gale_Encyclopedia_of_Medicine_5th_Ed.pdf"
OUTPUT_DIR      = "./output_datasets"
TOTAL_PAGES     = 6181
CONTENT_START   = 45       # First content page (after TOC/preface)
CONTENT_END     = 6100     # Last content page
BATCH_SIZE      = 100      # Pages processed per batch

# All known section headers in encyclopedia entries
ALL_SECTION_HEADERS = [
    "Definition", "Description", "Demographics", "Causes",
    "Causes and symptoms", "Causes & symptoms", "Risk factors",
    "Symptoms", "Signs and symptoms", "Pathophysiology",
    "Diagnosis", "Examination", "Tests", "Treatment",
    "Alternative treatment", "Prognosis", "Prevention",
    "Resources", "KEY TERMS", "Aftercare",
    "Complications", "Morbidity and mortality",
]

# Body system keyword mapping for automatic classification
BODY_SYSTEM_KEYWORDS = {
    "cardiovascular":    ["heart", "cardiac", "vascular", "artery", "vein",
                          "blood pressure", "coronary", "aorta"],
    "respiratory":       ["lung", "pulmonary", "bronch", "trachea",
                          "respiratory", "asthma", "pleura"],
    "neurological":      ["brain", "nerve", "neuron", "spinal", "neurolog",
                          "seizure", "cognitive", "cerebral"],
    "gastrointestinal":  ["stomach", "intestin", "bowel", "colon", "liver",
                          "pancreas", "digest", "esophag"],
    "musculoskeletal":   ["bone", "muscle", "joint", "arthritis", "tendon",
                          "ligament", "cartilage", "skeletal"],
    "endocrine":         ["hormone", "thyroid", "adrenal", "diabetes",
                          "insulin", "gland", "pituitary"],
    "immunological":     ["immune", "antibody", "autoimmune", "allerg",
                          "inflammatory", "lymphocyte"],
    "dermatological":    ["skin", "dermat", "rash", "lesion", "epiderm",
                          "melanoma", "keratin"],
    "renal":             ["kidney", "renal", "urinary", "bladder", "nephro",
                          "glomerul", "ureter"],
    "reproductive":      ["uterus", "ovarian", "testicular", "reproductive",
                          "cervical", "prostate", "endometri"],
    "psychiatric":       ["mental", "psychiatric", "anxiety", "depression",
                          "psycho", "mood", "schizophrenia"],
    "infectious":        ["bacteria", "virus", "fungal", "parasite",
                          "infection", "pathogen", "microbial"],
    "hematological":     ["blood", "anemia", "platelet", "leukemia",
                          "lymphoma", "hemato", "erythrocyte"],
    "oncological":       ["cancer", "tumor", "malignant", "carcinoma",
                          "sarcoma", "neoplasm", "metastasis"],
    "ophthalmological":  ["eye", "vision", "retina", "cornea", "ophthalm",
                          "glaucoma", "cataract"],
}

# Severity inference keywords
SEVERITY_KEYWORDS = {
    "Critical": ["fatal", "life-threatening", "death", "mortality",
                 "lethal", "terminal", "emergency"],
    "Severe":   ["severe", "serious", "major", "significant disability",
                 "chronic pain", "debilitating"],
    "Moderate": ["moderate", "treatable", "managed with treatment",
                 "manageable"],
    "Mild":     ["mild", "minor", "self-limiting", "resolve on their own",
                 "benign", "short-lived"],
}

# Standard symptom vocabulary (65 terms)
STANDARD_SYMPTOMS = [
    "pain", "fever", "fatigue", "nausea", "vomiting", "diarrhea",
    "headache", "dizziness", "cough", "rash", "swelling", "weakness",
    "numbness", "bleeding", "inflammation", "itching", "burning",
    "cramps", "breathlessness", "palpitations", "seizures", "tremors",
    "paralysis", "depression", "anxiety", "confusion", "memory loss",
    "weight loss", "weight gain", "jaundice", "edema", "anemia",
    "pallor", "cyanosis", "dyspnea", "tachycardia", "hypertension",
    "hypotension", "insomnia", "loss of appetite", "constipation",
    "urinary frequency", "hematuria", "alopecia", "pruritus",
    "erythema", "blurred vision", "hearing loss", "tinnitus", "vertigo",
    "syncope", "polyuria", "polydipsia", "chest pain", "abdominal pain",
    "joint pain", "back pain", "sore throat", "runny nose",
    "shortness of breath", "night sweats", "chills", "muscle aches",
    "skin lesions", "lymphadenopathy",
]

# ICD category mapping by entry type
ICD_CATEGORY_MAP = {
    "Disease/Condition":    "Various",
    "Disease/Disorder":     "Various",
    "Cancer/Oncological":   "C00-D49",
    "Diagnostic Test":      "Z-codes",
    "Surgical Procedure":   "Z-codes",
    "Drug/Medication":      "Various",
    "Nutrition/Diet":       "E40-E68",
    "Alternative Therapy":  "Z-codes",
}


# =============================================================================
# STEP 1 — PDF TEXT EXTRACTION
# =============================================================================

def extract_text_from_pages(pdf_path: str, start: int, end: int) -> str:
    """
    Extract raw text from a page range using pdftotext (poppler).
    Returns plain text string.
    """
    result = subprocess.run(
        ["pdftotext", "-f", str(start), "-l", str(end), pdf_path, "-"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed on pages {start}-{end}: {result.stderr}")
    return result.stdout


def clean_extracted_text(text: str) -> str:
    """
    Remove PDF artefacts, boilerplate headers, OCR noise, and
    page-break characters from extracted text.
    """
    # Remove publisher watermarks
    text = re.sub(r"Presented By:.*?\n", "", text)
    text = re.sub(r"WWW\.BOOKBAZ\.IR", "", text)

    # Remove encyclopedia header repeated on each page
    text = re.sub(
        r"G\s*A?\s*L?\s*E?\s+EN?\s*C?\s*Y?\s*C?\s*L?\s*O?\s*P?\s*E?\s*D?\s*I?\s*A?"
        r"\s+O?\s*F\s+M\s*E\s*D\s*I\s*C\s*I\s*N\s*E.*?EDITION",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalise KEY TERMS header (sometimes spaced by OCR)
    text = re.sub(r"KE\s*Y\s+T\s*E?\s*R\s*M\s*S?\b", "KEY TERMS", text)

    # Replace form-feed (page break) with newline
    text = text.replace("\f", "\n")

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


# =============================================================================
# STEP 2 — ENTRY DETECTION & SECTION PARSING
# =============================================================================

def find_entry_titles_and_positions(text: str) -> list[dict]:
    """
    Locate every encyclopedia entry in the cleaned text block.

    Each entry starts with a title line followed by a 'Definition' section.
    Strategy: find all 'Definition' anchors, then look backwards ~400 chars
    to identify the last clean non-boilerplate line as the title.

    Returns list of dicts: {title, content_start, content_end}
    """
    definition_matches = list(re.finditer(r"\nDefinition\n", text))
    entries = []

    for i, match in enumerate(definition_matches):
        pos = match.start()

        # ── Find title by looking backwards from the Definition anchor ──
        lookback = text[max(0, pos - 500) : pos]
        lines = [l.strip() for l in lookback.split("\n") if l.strip()]
        clean_lines = [
            l for l in lines
            if 3 < len(l) < 100
            and not l.isdigit()
            and not re.match(r"^\d+$", l)
            and "Presented By" not in l
            and "BOOKBAZ" not in l
            and "GALE" not in l.upper()[:10]
            and not re.search(r"[A-Z]\s+[A-Z]\s+[A-Z]", l)  # OCR spaced text
        ]
        if not clean_lines:
            continue
        title = clean_lines[-1]

        # ── Determine content block end ──
        next_pos = (
            definition_matches[i + 1].start()
            if i + 1 < len(definition_matches)
            else min(pos + 6000, len(text))
        )
        content = text[pos:next_pos]

        entries.append({"title": title, "content": content})

    return entries


def extract_section(content: str, section_names: list[str]) -> str:
    """
    Extract text between a matched section header and the next known header.
    Returns empty string if section not found.
    """
    escaped_all = "|".join(re.escape(h) for h in ALL_SECTION_HEADERS)
    for header in section_names:
        pattern = (
            rf"(?:^|\n){re.escape(header)}\s*\n"
            rf"(.*?)"
            rf"(?=\n(?:{escaped_all})|\Z)"
        )
        m = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if m:
            # Collapse whitespace and cap length
            return " ".join(m.group(1).split())[:1500]
    return ""


def extract_key_terms(content: str) -> str:
    """
    Extract medical term names from the KEY TERMS section.
    Terms are identified as short lines preceding an em-dash definition.
    """
    m = re.search(
        r"KEY TERMS\s*\n(.*?)(?=\n(?:Definition|Description|Causes|"
        r"Treatment|Prognosis|Prevention|Resources)|\Z)",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return ""
    block = m.group(1)
    terms = re.findall(r"^([A-Z][A-Za-z\s\-]+?)(?:—|–|-)", block, re.MULTILINE)
    return "; ".join(t.strip() for t in terms[:10] if len(t.strip()) < 50)


# =============================================================================
# STEP 3 — FEATURE ENGINEERING
# =============================================================================

def classify_entry_type(title: str, definition: str, description: str) -> str:
    """
    Rule-based classification of each encyclopedia entry into one of:
    Diagnostic Test | Surgical Procedure | Drug/Medication |
    Cancer/Oncological | Disease/Disorder | Nutrition/Diet |
    Alternative Therapy | Disease/Condition
    """
    t = title.lower()
    combined = (definition + " " + description).lower()

    if any(x in t for x in ["test", "scan", "biopsy", "assay", "x-ray",
                              "imaging", "ultrasound", "mri"]):
        return "Diagnostic Test"

    if any(x in t for x in ["surgery", "surgical", "ectomy", "ostomy",
                              "plasty", "otomy", "oscopy"]):
        return "Surgical Procedure"

    if any(x in t for x in ["drug", "medication", "antibiotic"]):
        if any(x in combined for x in ["dosage", "prescri", "tablet",
                                        "capsule", "injection"]):
            return "Drug/Medication"

    if any(x in t for x in ["cancer", "tumor", "carcinoma", "sarcoma",
                              "leukemia", "lymphoma"]):
        return "Cancer/Oncological"

    if any(x in t for x in ["nutrition", "diet", "vitamin", "mineral"]):
        return "Nutrition/Diet"

    if any(x in t for x in ["therapy", "acupuncture", "massage",
                              "chiropractic", "homeopathic"]):
        return "Alternative Therapy"

    if any(x in t for x in ["syndrome", "disorder", "disease", "infection",
                              "deficiency", "failure"]):
        return "Disease/Disorder"

    return "Disease/Condition"


def infer_body_systems(text: str) -> str:
    """
    Detect which body systems are involved using keyword matching.
    Returns semicolon-separated list of matched systems.
    """
    lower = text.lower()
    matched = [
        system
        for system, keywords in BODY_SYSTEM_KEYWORDS.items()
        if any(kw in lower for kw in keywords)
    ]
    return "; ".join(matched[:4])


def infer_severity(prognosis: str, description: str, definition: str) -> str:
    """
    Infer severity level from prognosis and description text.
    """
    text = " ".join([prognosis, description, definition]).lower()
    for level, keywords in SEVERITY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return level
    return "Unknown"


def extract_age_groups(text: str) -> str:
    """Detect affected age groups from description text."""
    lower = text.lower()
    groups = []
    if any(x in lower for x in ["infant", "newborn", "neonate"]):
        groups.append("Infants")
    if any(x in lower for x in ["children", "childhood", "pediatric"]):
        groups.append("Children")
    if any(x in lower for x in ["adolescent", "teen", "puberty"]):
        groups.append("Adolescents")
    if any(x in lower for x in ["adult", "elderly", "geriatric", "aged 65"]):
        groups.append("Adults/Elderly")
    return "; ".join(groups) if groups else "All ages"


def infer_inheritance(causes: str, description: str) -> str:
    """Detect genetic/inheritance pattern from text."""
    text = (causes + " " + description).lower()
    if "autosomal recessive" in text: return "Autosomal Recessive"
    if "autosomal dominant" in text:  return "Autosomal Dominant"
    if "x-linked" in text:            return "X-linked"
    if any(x in text for x in ["genetic", "hereditary", "inherited"]):
        return "Genetic/Hereditary"
    if "congenital" in text:          return "Congenital"
    return ""


def is_contagious(causes: str, description: str, prevention: str) -> str:
    """Detect if condition is contagious/transmissible."""
    text = " ".join([causes, description, prevention]).lower()
    if any(x in text for x in ["contagious", "transmit", "spread from person",
                                 "infectious", "contact precaution"]):
        return "Yes"
    return "No"


def extract_symptom_vector(symptom_text: str) -> list[int]:
    """
    Build a binary symptom presence vector over the 65-term standard vocabulary.
    Returns list of 0/1 integers aligned to STANDARD_SYMPTOMS order.
    """
    lower = symptom_text.lower()
    return [1 if sym in lower else 0 for sym in STANDARD_SYMPTOMS]


# =============================================================================
# STEP 4 — PARSE SINGLE ENTRY INTO STRUCTURED RECORD
# =============================================================================

def parse_entry(title: str, content: str) -> dict:
    """
    Given an entry title and its full text content block,
    extract all structured fields and return as a flat dict.
    """
    # ── Section extraction ──
    definition  = extract_section(content, ["Definition"])
    description = extract_section(content, ["Description", "Demographics"])
    causes      = extract_section(content, ["Causes and symptoms",
                                             "Causes & symptoms",
                                             "Causes", "Risk factors"])
    symptoms    = extract_section(content, ["Symptoms", "Signs and symptoms"])
    diagnosis   = extract_section(content, ["Diagnosis", "Examination", "Tests"])
    treatment   = extract_section(content, ["Treatment", "Alternative treatment"])
    prognosis   = extract_section(content, ["Prognosis"])
    prevention  = extract_section(content, ["Prevention"])
    complications = extract_section(content, ["Complications", "Aftercare"])
    key_terms   = extract_key_terms(content)

    # If no separate symptoms section, try to mine from causes text
    if not symptoms and causes:
        sym_matches = re.findall(
            r"\b(?:symptoms?|signs?)\s+(?:include|such as|are|of)[:\s]+([^.]+)",
            causes, re.IGNORECASE,
        )
        symptoms = "; ".join(sym_matches[:3])

    # ── Feature engineering ──
    combined_text = " ".join([definition, description, causes])
    entry_type    = classify_entry_type(title, definition, description)
    body_systems  = infer_body_systems(combined_text)
    severity      = infer_severity(prognosis, description, definition)
    age_groups    = extract_age_groups(description + " " + definition)
    inheritance   = infer_inheritance(causes, description)
    contagious    = is_contagious(causes, description, prevention)
    icd_category  = ICD_CATEGORY_MAP.get(entry_type, "Various")

    return {
        "entry_name":          title,
        "entry_type":          entry_type,
        "icd_category":        icd_category,
        "body_systems":        body_systems,
        "severity_level":      severity,
        "age_groups_affected": age_groups,
        "inheritance_pattern": inheritance,
        "is_contagious":       contagious,
        "definition":          definition[:500],
        "description":         description[:600],
        "causes":              causes[:600],
        "symptoms":            symptoms[:400],
        "diagnosis":           diagnosis[:500],
        "treatment":           treatment[:600],
        "prognosis":           prognosis[:300],
        "prevention":          prevention[:300],
        "complications":       complications[:300],
        "key_terms":           key_terms[:400],
    }


# =============================================================================
# STEP 5 — FULL PDF PROCESSING LOOP
# =============================================================================

def process_pdf(pdf_path: str) -> list[dict]:
    """
    Iterate through the PDF in BATCH_SIZE page batches.
    For each batch: extract text → clean → find entries → parse.
    Returns deduplicated list of all parsed entry dicts.
    """
    all_entries = []
    seen_titles = set()

    batches = list(range(CONTENT_START, CONTENT_END, BATCH_SIZE))
    print(f"\n[Pipeline] Processing {TOTAL_PAGES}-page PDF in "
          f"{len(batches)} batches of {BATCH_SIZE} pages...\n")

    for start in tqdm(batches, desc="Extracting pages", unit="batch"):
        end = min(start + BATCH_SIZE - 1, CONTENT_END)

        # 1. Extract raw text
        raw_text = extract_text_from_pages(pdf_path, start, end)

        # 2. Clean artefacts
        clean_text = clean_extracted_text(raw_text)

        # 3. Detect entry positions
        entry_positions = find_entry_titles_and_positions(clean_text)

        # 4. Parse each entry
        for ep in entry_positions:
            title = ep["title"].strip()
            title_key = title.lower()

            # Skip duplicates (entries spanning batch boundaries)
            if title_key in seen_titles:
                continue
            # Skip OCR garbage titles
            if re.search(r"[A-Z]\s+[A-Z]\s+[A-Z]", title):
                continue
            if len(title) > 80 or len(title) < 3:
                continue

            record = parse_entry(title, ep["content"])

            # Only keep entries with at least a definition or description
            if record["definition"] or record["description"]:
                all_entries.append(record)
                seen_titles.add(title_key)

    print(f"\n[Pipeline] Extracted {len(all_entries)} unique entries.\n")
    return all_entries


# =============================================================================
# DATASET 1 — MASTER ENCYCLOPEDIA ENTRIES
# =============================================================================

DATASET1_FIELDS = [
    "entry_name", "entry_type", "icd_category", "body_systems",
    "severity_level", "age_groups_affected", "inheritance_pattern",
    "is_contagious", "definition", "description", "causes", "symptoms",
    "diagnosis", "treatment", "prognosis", "prevention",
    "complications", "key_terms",
]

def save_dataset1(entries: list[dict], output_dir: str) -> str:
    path = os.path.join(output_dir, "01_medical_encyclopedia_entries.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET1_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    print(f"[Dataset 1] Saved {len(entries)} entries → {path}")
    return path


# =============================================================================
# DATASET 2 — KNOWLEDGE GRAPH NODES
# =============================================================================

def build_nodes(entries: list[dict]) -> tuple[list[dict], dict]:
    """
    Build typed node records for the knowledge graph.

    Node types created:
      Disease       — from disease/disorder/cancer entries
      Symptom       — from STANDARD_SYMPTOMS vocabulary
      BodySystem    — from BODY_SYSTEM_KEYWORDS keys
      Procedure     — from surgical procedure entries
      Drug          — from drug/medication entries
      DiagnosticTest— from diagnostic test entries
    """
    nodes = []
    node_id_counter = [1]  # mutable for use in nested helpers
    node_lookup = {}       # label → node_id, for edge building

    def next_id(prefix):
        nid = f"{prefix}{node_id_counter[0]:05d}"
        node_id_counter[0] += 1
        return nid

    # ── Disease nodes ──
    disease_types = {"Disease/Condition", "Disease/Disorder",
                     "Cancer/Oncological"}
    for e in entries:
        if e["entry_type"] not in disease_types:
            continue
        nid = next_id("D")
        label = e["entry_name"]
        nodes.append({
            "node_id":      nid,
            "node_type":    "Disease",
            "label":        label,
            "definition":   e["definition"][:300],
            "body_systems": e["body_systems"],
            "severity":     e["severity_level"],
            "age_groups":   e["age_groups_affected"],
            "icd_category": e["icd_category"],
            "is_contagious":e["is_contagious"],
            "inheritance":  e["inheritance_pattern"],
        })
        node_lookup[("Disease", label.lower())] = nid

    # ── Symptom nodes ──
    for sym in STANDARD_SYMPTOMS:
        nid = next_id("S")
        nodes.append({
            "node_id":      nid,
            "node_type":    "Symptom",
            "label":        sym.title(),
            "definition":   f"Clinical manifestation: {sym}",
            "body_systems": "",
            "severity":     "",
            "age_groups":   "",
            "icd_category": "R-codes",
            "is_contagious":"",
            "inheritance":  "",
        })
        node_lookup[("Symptom", sym.lower())] = nid

    # ── Body system nodes ──
    for system in BODY_SYSTEM_KEYWORDS:
        nid = next_id("BS")
        nodes.append({
            "node_id":      nid,
            "node_type":    "BodySystem",
            "label":        system.title(),
            "definition":   f"Body system: {system}",
            "body_systems": system,
            "severity":     "",
            "age_groups":   "",
            "icd_category": "",
            "is_contagious":"",
            "inheritance":  "",
        })
        node_lookup[("BodySystem", system.lower())] = nid

    # ── Procedure nodes ──
    for e in entries:
        if e["entry_type"] not in {"Surgical Procedure", "Alternative Therapy"}:
            continue
        nid = next_id("P")
        label = e["entry_name"]
        nodes.append({
            "node_id":      nid,
            "node_type":    "Procedure",
            "label":        label,
            "definition":   e["definition"][:300],
            "body_systems": e["body_systems"],
            "severity":     "",
            "age_groups":   e["age_groups_affected"],
            "icd_category": e["icd_category"],
            "is_contagious":"",
            "inheritance":  "",
        })
        node_lookup[("Procedure", label.lower())] = nid

    # ── Drug nodes ──
    for e in entries:
        if e["entry_type"] != "Drug/Medication":
            continue
        nid = next_id("DR")
        label = e["entry_name"]
        nodes.append({
            "node_id":      nid,
            "node_type":    "Drug",
            "label":        label,
            "definition":   e["definition"][:300],
            "body_systems": e["body_systems"],
            "severity":     "",
            "age_groups":   "",
            "icd_category": e["icd_category"],
            "is_contagious":"",
            "inheritance":  "",
        })
        node_lookup[("Drug", label.lower())] = nid

    # ── Diagnostic Test nodes ──
    for e in entries:
        if e["entry_type"] != "Diagnostic Test":
            continue
        nid = next_id("T")
        label = e["entry_name"]
        nodes.append({
            "node_id":      nid,
            "node_type":    "DiagnosticTest",
            "label":        label,
            "definition":   e["definition"][:300],
            "body_systems": e["body_systems"],
            "severity":     "",
            "age_groups":   "",
            "icd_category": e["icd_category"],
            "is_contagious":"",
            "inheritance":  "",
        })
        node_lookup[("DiagnosticTest", label.lower())] = nid

    print(f"[Dataset 2] Built {len(nodes)} nodes across "
          f"{len(set(n['node_type'] for n in nodes))} types.")
    return nodes, node_lookup


NODE_FIELDS = [
    "node_id", "node_type", "label", "definition", "body_systems",
    "severity", "age_groups", "icd_category", "is_contagious", "inheritance",
]

def save_dataset2(nodes: list[dict], output_dir: str) -> str:
    path = os.path.join(output_dir, "02_knowledge_graph_nodes.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NODE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(nodes)
    print(f"[Dataset 2] Saved {len(nodes)} nodes → {path}")
    return path


# =============================================================================
# DATASET 3 — KNOWLEDGE GRAPH EDGES / RELATIONSHIPS
# =============================================================================

def build_edges(entries: list[dict],
                nodes: list[dict],
                node_lookup: dict) -> list[dict]:
    """
    Derive three relationship types between nodes:

      AFFECTS        Disease  → BodySystem   (via body_systems field)
      HAS_SYMPTOM    Disease  → Symptom      (via symptom text matching)
      TREATS         Procedure→ Disease      (via shared body system overlap)
    """
    edges = []
    edge_id = 1

    # Build quick-access dicts
    disease_entries    = {e["entry_name"].lower(): e for e in entries
                          if e["entry_type"] in {"Disease/Condition",
                                                  "Disease/Disorder",
                                                  "Cancer/Oncological"}}
    procedure_entries  = [e for e in entries
                          if e["entry_type"] in {"Surgical Procedure",
                                                  "Alternative Therapy"}]

    # ── AFFECTS: Disease → BodySystem ──
    for node in nodes:
        if node["node_type"] != "Disease":
            continue
        disease_key = node["label"].lower()
        entry = disease_entries.get(disease_key, {})
        if not entry:
            continue
        for system in entry.get("body_systems", "").split("; "):
            system = system.strip().lower()
            if not system:
                continue
            target_id = node_lookup.get(("BodySystem", system))
            if target_id:
                edges.append({
                    "edge_id":        f"E{edge_id:06d}",
                    "source_node_id": node["node_id"],
                    "source_label":   node["label"],
                    "relationship":   "AFFECTS",
                    "target_node_id": target_id,
                    "target_label":   system.title(),
                    "weight":         1.0,
                    "evidence":       "body_system_classification",
                })
                edge_id += 1

    # ── HAS_SYMPTOM: Disease → Symptom ──
    for node in nodes:
        if node["node_type"] != "Disease":
            continue
        disease_key = node["label"].lower()
        entry = disease_entries.get(disease_key, {})
        if not entry:
            continue
        symptom_text = " ".join([
            entry.get("symptoms", ""),
            entry.get("causes", ""),
            entry.get("description", ""),
        ]).lower()

        for sym in STANDARD_SYMPTOMS:
            if sym in symptom_text:
                target_id = node_lookup.get(("Symptom", sym.lower()))
                if target_id:
                    edges.append({
                        "edge_id":        f"E{edge_id:06d}",
                        "source_node_id": node["node_id"],
                        "source_label":   node["label"],
                        "relationship":   "HAS_SYMPTOM",
                        "target_node_id": target_id,
                        "target_label":   sym.title(),
                        "weight":         1.0,
                        "evidence":       "symptom_text_matching",
                    })
                    edge_id += 1

    # ── TREATS: Procedure → Disease (body system overlap) ──
    for proc_entry in procedure_entries:
        proc_id = node_lookup.get(("Procedure", proc_entry["entry_name"].lower()))
        if not proc_id:
            continue
        proc_systems = set(
            s.strip().lower()
            for s in proc_entry["body_systems"].split("; ")
            if s.strip()
        )
        for dis_name, dis_entry in disease_entries.items():
            dis_systems = set(
                s.strip().lower()
                for s in dis_entry["body_systems"].split("; ")
                if s.strip()
            )
            if proc_systems & dis_systems:  # shared body system
                dis_id = node_lookup.get(("Disease", dis_name))
                if dis_id:
                    edges.append({
                        "edge_id":        f"E{edge_id:06d}",
                        "source_node_id": proc_id,
                        "source_label":   proc_entry["entry_name"],
                        "relationship":   "TREATS",
                        "target_node_id": dis_id,
                        "target_label":   dis_entry["entry_name"],
                        "weight":         0.5,
                        "evidence":       "body_system_overlap",
                    })
                    edge_id += 1
            if edge_id > 55000:
                break

    print(f"[Dataset 3] Built {len(edges)} edges "
          f"(AFFECTS + HAS_SYMPTOM + TREATS).")
    return edges


EDGE_FIELDS = [
    "edge_id", "source_node_id", "source_label", "relationship",
    "target_node_id", "target_label", "weight", "evidence",
]

def save_dataset3(edges: list[dict], output_dir: str) -> str:
    path = os.path.join(output_dir, "03_knowledge_graph_edges.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EDGE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(edges)
    print(f"[Dataset 3] Saved {len(edges)} edges → {path}")
    return path


# =============================================================================
# DATASET 4 — DISEASE × SYMPTOM BINARY MATRIX
# =============================================================================

def build_symptom_matrix(entries: list[dict]) -> list[dict]:
    """
    Build a binary presence/absence matrix:
      Rows    = disease entries
      Columns = STANDARD_SYMPTOMS (65 terms)
      Values  = 1 if symptom mentioned in entry text, else 0
    """
    disease_types = {"Disease/Condition", "Disease/Disorder",
                     "Cancer/Oncological"}
    matrix_rows = []

    for entry in entries:
        if entry["entry_type"] not in disease_types:
            continue
        symptom_text = " ".join([
            entry.get("symptoms", ""),
            entry.get("causes", ""),
        ])
        vec = extract_symptom_vector(symptom_text)
        row = {
            "disease_name": entry["entry_name"],
            "entry_type":   entry["entry_type"],
        }
        for sym, val in zip(STANDARD_SYMPTOMS, vec):
            row[sym] = val
        matrix_rows.append(row)

    print(f"[Dataset 4] Built matrix: "
          f"{len(matrix_rows)} diseases × {len(STANDARD_SYMPTOMS)} symptoms.")
    return matrix_rows


def save_dataset4(matrix: list[dict], output_dir: str) -> str:
    path = os.path.join(output_dir, "04_disease_symptom_matrix.csv")
    fields = ["disease_name", "entry_type"] + STANDARD_SYMPTOMS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(matrix)
    print(f"[Dataset 4] Saved {len(matrix)} rows → {path}")
    return path


# =============================================================================
# DATASET 5 — MEDICAL GLOSSARY
# =============================================================================

def build_glossary(entries: list[dict]) -> list[dict]:
    """
    Aggregate unique medical terms from KEY TERMS sections across all entries.
    For each term, record frequency and which entries it appears in.
    """
    term_index = defaultdict(lambda: {"appears_in": [], "entry_types": set()})

    for entry in entries:
        raw_terms = entry.get("key_terms", "")
        if not raw_terms:
            continue
        for term in raw_terms.split("; "):
            term = term.strip()
            if term and len(term) > 2:
                term_index[term]["appears_in"].append(entry["entry_name"])
                term_index[term]["entry_types"].add(entry["entry_type"])

    glossary = []
    for term, info in sorted(term_index.items()):
        glossary.append({
            "term":              term,
            "frequency":         len(info["appears_in"]),
            "associated_entries": "; ".join(info["appears_in"][:10]),
            "entry_types":        "; ".join(info["entry_types"]),
        })

    glossary.sort(key=lambda x: x["frequency"], reverse=True)
    print(f"[Dataset 5] Built glossary: {len(glossary)} unique terms.")
    return glossary


def save_dataset5(glossary: list[dict], output_dir: str) -> str:
    path = os.path.join(output_dir, "05_medical_glossary.csv")
    fields = ["term", "frequency", "associated_entries", "entry_types"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(glossary)
    print(f"[Dataset 5] Saved {len(glossary)} terms → {path}")
    return path


# =============================================================================
# PIPELINE SUMMARY REPORT
# =============================================================================

def print_summary(entries, nodes, edges, matrix, glossary):
    print("\n" + "=" * 62)
    print("  PIPELINE COMPLETE — SUMMARY REPORT")
    print("=" * 62)

    # Entry type distribution
    type_counts = Counter(e["entry_type"] for e in entries)
    print("\n  Entry type distribution:")
    for etype, cnt in type_counts.most_common():
        print(f"    {etype:<30} {cnt:>5}")

    # Node type distribution
    node_counts = Counter(n["node_type"] for n in nodes)
    print("\n  Node type distribution:")
    for ntype, cnt in node_counts.most_common():
        print(f"    {ntype:<20} {cnt:>5}")

    # Edge type distribution
    edge_counts = Counter(e["relationship"] for e in edges)
    print("\n  Relationship type distribution:")
    for rel, cnt in edge_counts.most_common():
        print(f"    {rel:<20} {cnt:>6}")

    # Body system coverage
    system_counts = Counter()
    for e in entries:
        for s in e["body_systems"].split("; "):
            if s.strip():
                system_counts[s.strip()] += 1
    print("\n  Top body systems:")
    for sys, cnt in system_counts.most_common(8):
        print(f"    {sys:<22} {cnt:>5}")

    # Dataset sizes
    print(f"\n  Dataset sizes:")
    print(f"    01_medical_encyclopedia_entries  {len(entries):>6} rows")
    print(f"    02_knowledge_graph_nodes         {len(nodes):>6} rows")
    print(f"    03_knowledge_graph_edges         {len(edges):>6} rows")
    print(f"    04_disease_symptom_matrix        {len(matrix):>6} rows")
    print(f"    05_medical_glossary              {len(glossary):>6} rows")
    print("=" * 62 + "\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Medical Knowledge Graph Dataset Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdf", default=PDF_PATH,
        help="Path to the Gale Encyclopedia of Medicine PDF",
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help="Directory to save output CSV files",
    )
    parser.add_argument(
        "--start", type=int, default=CONTENT_START,
        help="First content page to process (default: 45)",
    )
    parser.add_argument(
        "--end", type=int, default=CONTENT_END,
        help="Last content page to process (default: 6100)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Medical Knowledge Graph Dataset Pipeline")
    print(f"  Source: Gale Encyclopedia of Medicine, 5th Ed. (2015)")
    print(f"  PDF:    {args.pdf}")
    print(f"  Pages:  {args.start} → {args.end}")
    print(f"  Output: {args.output}")
    print(f"{'='*62}\n")

    # ── Run pipeline stages ──

    # Stage 1: Extract and parse all entries from PDF
    entries = process_pdf(args.pdf)

    # Stage 2: Save Dataset 1
    save_dataset1(entries, args.output)

    # Stage 3: Build knowledge graph nodes
    nodes, node_lookup = build_nodes(entries)
    save_dataset2(nodes, args.output)

    # Stage 4: Build knowledge graph edges
    edges = build_edges(entries, nodes, node_lookup)
    save_dataset3(edges, args.output)

    # Stage 5: Build disease-symptom matrix
    matrix = build_symptom_matrix(entries)
    save_dataset4(matrix, args.output)

    # Stage 6: Build medical glossary
    glossary = build_glossary(entries)
    save_dataset5(glossary, args.output)

    # Stage 7: Print summary
    print_summary(entries, nodes, edges, matrix, glossary)


if __name__ == "__main__":
    main()
