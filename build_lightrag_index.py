#!/usr/bin/env python3
"""
build_lightrag_index.py — Build a LightRAG index from the MediBot encyclopedia corpus.

Indexes 01_medical_encyclopedia_entries.csv into lightrag_storage/ for use as a
graph+vector RAG baseline against MediBot's hand-built knowledge graph.

Usage:
    python build_lightrag_index.py
    python build_lightrag_index.py --limit 50          # quick smoke test
    python build_lightrag_index.py --rebuild           # wipe and rebuild
    python build_lightrag_index.py --resume            # continue partial index
    python build_lightrag_index.py --batch-size 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from lightrag_utils import (
    CREDITS_FLAG_PATH,
    DEFAULT_DATASET_DIR,
    DEFAULT_STORAGE_DIR,
    CreditsExhaustedError,
    create_lightrag,
    load_corpus_documents,
    reset_storage,
    storage_is_ready,
)

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "lightrag_index_manifest.json"

# Exit code used when the LLM provider runs out of credits (402).
# run_lightrag_pipeline.py treats this as "stop cleanly, wait for new key".
EXIT_CREDITS_EXHAUSTED = 42


def load_processed_doc_ids(storage_dir: Path) -> set[str]:
    status_path = storage_dir / "kv_store_doc_status.json"
    if not status_path.exists():
        return set()
    data = json.loads(status_path.read_text(encoding="utf-8"))
    return {doc_id for doc_id, meta in data.items() if meta.get("status") == "processed"}


async def build_index(
    dataset_dir: Path,
    storage_dir: Path,
    batch_size: int,
    limit: int | None,
    rebuild: bool,
    resume: bool,
) -> None:
    if rebuild:
        print(f"Removing existing index at {storage_dir} ...")
        reset_storage(storage_dir)
    elif storage_is_ready(storage_dir) and not resume:
        print(f"Index already exists at {storage_dir}. Use --rebuild or --resume.")
        return

    docs, doc_ids, names = load_corpus_documents(dataset_dir, limit=limit)
    if not docs:
        raise RuntimeError("No encyclopedia documents found to index.")

    if resume:
        processed_ids = load_processed_doc_ids(storage_dir)
        pending = [
            (doc, doc_id, name)
            for doc, doc_id, name in zip(docs, doc_ids, names)
            if doc_id not in processed_ids
        ]
        docs = [row[0] for row in pending]
        doc_ids = [row[1] for row in pending]
        names = [row[2] for row in pending]
        print(
            f"Resuming index build: {len(processed_ids)} already processed, "
            f"{len(docs)} remaining ..."
        )
        if not docs:
            print("Nothing left to index.")
            return

    print(f"Preparing LightRAG index with {len(docs)} documents ...")
    # Clear any stale credits flag from a previous run before starting.
    CREDITS_FLAG_PATH.unlink(missing_ok=True)
    rag = await create_lightrag(storage_dir=storage_dir)

    total_batches = (len(docs) + batch_size - 1) // batch_size
    t0 = time.time()

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(docs))
        batch_docs = docs[start:end]
        batch_ids = doc_ids[start:end]
        batch_names = names[start:end]

        print(
            f"  Inserting batch {batch_idx + 1}/{total_batches} "
            f"({start + 1}-{end} of {len(docs)}) ..."
        )
        try:
            await rag.ainsert(batch_docs, ids=batch_ids)
        except CreditsExhaustedError as ex:
            print(f"\n*** CREDITS EXHAUSTED — stopping cleanly. ***\n{ex}")
            await rag.finalize_storages()
            sys.exit(EXIT_CREDITS_EXHAUSTED)

        # LightRAG swallows per-doc errors, so also check the sentinel flag.
        if CREDITS_FLAG_PATH.exists():
            detail = CREDITS_FLAG_PATH.read_text(encoding="utf-8")
            print(f"\n*** CREDITS EXHAUSTED — stopping cleanly after batch {batch_idx + 1}. ***")
            print(detail)
            print("Update OPENROUTER_API_KEY in .env, then re-run with --resume.")
            await rag.finalize_storages()
            sys.exit(EXIT_CREDITS_EXHAUSTED)

        preview = ", ".join(batch_names[:3])
        if len(batch_names) > 3:
            preview += ", ..."
        print(f"    Examples: {preview}")

    elapsed = time.time() - t0
    manifest = {
        "documents_indexed": len(load_processed_doc_ids(storage_dir)),
        "documents_target": len(load_corpus_documents(dataset_dir, limit=limit)[0]),
        "batch_size": batch_size,
        "dataset_dir": str(dataset_dir),
        "storage_dir": str(storage_dir),
        "elapsed_sec": round(elapsed, 2),
        "sample_entries": names[:10],
        "resumed": resume,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    await rag.finalize_storages()
    print(f"\nDone. Indexed {len(docs)} documents in {elapsed:.1f}s.")
    print(f"Total processed in storage: {len(load_processed_doc_ids(storage_dir))}")
    print(f"Manifest saved to {MANIFEST_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build LightRAG index from encyclopedia CSV.")
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--storage-dir", default=str(DEFAULT_STORAGE_DIR))
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="Index only first N valid entries.")
    ap.add_argument("--rebuild", action="store_true", help="Delete existing index and rebuild.")
    ap.add_argument("--resume", action="store_true", help="Continue indexing only unprocessed documents.")
    args = ap.parse_args()

    if args.rebuild and args.resume:
        print("ERROR: Use either --rebuild or --resume, not both.", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(
            build_index(
                dataset_dir=Path(args.dataset_dir),
                storage_dir=Path(args.storage_dir),
                batch_size=max(1, args.batch_size),
                limit=args.limit,
                rebuild=args.rebuild,
                resume=args.resume,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
