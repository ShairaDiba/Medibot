"""Rebuild kv_store_doc_status.json from actual index contents.

Marks every enc_* doc that exists in kv_store_full_docs.json as 'processed',
and leaves the rest absent (pipeline will treat them as new/pending).
"""
import json
from collections import Counter
from pathlib import Path

storage = Path("lightrag_storage")

# Docs actually indexed (full_docs is written after a doc completes)
full_docs = json.load(open(storage / "kv_store_full_docs.json", encoding="utf-8"))
print("Docs in full_docs store:", len(full_docs))

# Load the repaired status file (144 entries) to preserve its metadata shape
status = json.load(open(storage / "kv_store_doc_status.json", encoding="utf-8"))
print("Existing status entries:", len(status))

# A sample processed entry to use as a template
template = None
for v in status.values():
    if v.get("status") == "processed":
        template = v
        break

added = 0
for doc_id in full_docs:
    if doc_id not in status:
        entry = dict(template) if template else {"status": "processed"}
        entry["status"] = "processed"
        entry.pop("error_msg", None)
        entry["chunks_list"] = entry.get("chunks_list", [])
        status[doc_id] = entry
        added += 1

print("Added", added, "entries from full_docs")
print(Counter(v.get("status") for v in status.values()))

json.dump(
    status,
    open(storage / "kv_store_doc_status.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=2,
)
# Validate
json.load(open(storage / "kv_store_doc_status.json", encoding="utf-8"))
print("Written and validated OK. Total:", len(status))
