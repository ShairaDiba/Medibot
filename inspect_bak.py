import json
import re
from collections import Counter

raw = open(
    "lightrag_storage/kv_store_doc_status.corrupted.bak.json",
    encoding="utf-8",
    errors="replace",
).read()
print("File size:", len(raw))
print("Last 300 chars:", repr(raw[-300:]))
ids = re.findall(r'"(enc_[0-9]+)":', raw)
print("Unique doc IDs in file:", len(set(ids)))
