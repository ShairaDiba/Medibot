"""Repair kv_store_doc_status.json corrupted by concurrent writes.

Strategy: scan the raw text for complete per-doc JSON objects using a
brace-matching parser, keep the LAST occurrence of each doc id, and write
a clean JSON file. A backup of the corrupted file is saved first.
"""
import json
import re
import shutil
from pathlib import Path

STATUS = Path("lightrag_storage/kv_store_doc_status.json")
BACKUP = Path("lightrag_storage/kv_store_doc_status.corrupted.bak.json")

raw = STATUS.read_text(encoding="utf-8")
shutil.copy2(STATUS, BACKUP)
print(f"Backup saved: {BACKUP} ({len(raw)} bytes)")

# Find all top-level "doc-id": { ... } blocks with brace matching
docs = {}
# Match any key followed by an opening brace
for m in re.finditer(r'"([A-Za-z0-9_\-]+)":\s*\{', raw):
    doc_id = m.group(1)
    start = m.end() - 1  # position of '{'
    depth = 0
    i = start
    in_str = False
    esc = False
    while i < len(raw):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    block = raw[start : i + 1]
                    try:
                        docs[doc_id] = json.loads(block)
                    except json.JSONDecodeError:
                        pass  # truncated/garbled block — skip
                    break
        i += 1

print(f"Recovered {len(docs)} doc entries")
from collections import Counter
print(Counter(v.get("status") for v in docs.values()))

STATUS.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
print("Repaired file written.")

# Validate
json.loads(STATUS.read_text(encoding="utf-8"))
print("Validation OK")
