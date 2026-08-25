"""Fix the doc status file by recovering valid entries and correcting counts."""
import json
import re
from collections import Counter
from pathlib import Path

STORAGE = Path("lightrag_storage")
STATUS = STORAGE / "kv_store_doc_status.json"
BACKUP = STORAGE / "kv_store_doc_status.corrupted.bak.json"
LOG = Path("lightrag_pipeline_console.log")

# Step 1: Recover valid entries from corrupted backup
print("=== Step 1: Recovering from backup ===")
with open(BACKUP, encoding="utf-8") as f:
    raw = f.read()

docs = {}
i = 0
while True:
    # Find next doc key - use [0-9] instead of \d to avoid PowerShell issues
    m = re.search(r'"(enc_[0-9]+)":\s*\{', raw[i:])
    if not m:
        break
    doc_id = m.group(1)
    start = i + m.end() - 1  # position of opening brace

    # Find matching closing brace
    depth = 1
    j = start + 1
    while j < len(raw) and depth > 0:
        if raw[j] == "{":
            depth += 1
        elif raw[j] == "}":
            depth -= 1
        j += 1

    if depth == 0:
        try:
            doc_json = raw[start:j]
            doc_data = json.loads(doc_json)
            docs[doc_id] = doc_data
        except json.JSONDecodeError:
            pass  # Skip malformed entries
        i = j
    else:
        break

print(f"Recovered {len(docs)} valid entries")
c = Counter(v.get("status") for v in docs.values())
print(f"Recovered status: {dict(c)}")

# Step 2: Find actually processed docs from log
print("\n=== Step 2: Finding processed docs from log ===")
with open(LOG, encoding="utf-8") as f:
    log = f.read()

# Find all doc IDs that were processed
processed_ids = set(re.findall(r"Processing d-id: (enc_[0-9]+)", log))
print(f"Doc IDs processed in current run: {len(processed_ids)}")

# Step 3: Update status - mark processed docs and clear old errors
print("\n=== Step 3: Updating status ===")
updated = 0
for doc_id in processed_ids:
    if doc_id in docs:
        if docs[doc_id].get("status") != "processed":
            docs[doc_id]["status"] = "processed"
            docs[doc_id].pop("error_msg", None)
            updated += 1

print(f"Updated {updated} entries to 'processed'")

# Step 4: Reset old failed entries (from 402 errors) to pending
print("\n=== Step 4: Resetting old failed entries ===")
reset = 0
for doc_id, doc in docs.items():
    if doc.get("status") == "failed":
        error = doc.get("error_msg", "")
        # Reset if it's an old 402 error or no error message
        if "402" in error or "Insufficient credits" in error or not error:
            doc["status"] = "pending"
            doc.pop("error_msg", None)
            reset += 1

print(f"Reset {reset} old failed entries to 'pending'")

# Step 5: Write clean file
print("\n=== Step 5: Writing clean file ===")
with open(STATUS, "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2, ensure_ascii=False)

# Verify
with open(STATUS, encoding="utf-8") as f:
    json.load(f)  # Validate

c = Counter(v.get("status") for v in docs.values())
print(f"\n=== Final status ===")
print(f"Total: {len(docs)}")
print(f"Status: {dict(c)}")
print("Validation: OK")
