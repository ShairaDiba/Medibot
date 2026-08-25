#!/usr/bin/env python3
"""Shared helpers for LightRAG baseline indexing and evaluation."""

from __future__ import annotations

import os
import re
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "final_dataset"
DEFAULT_STORAGE_DIR = ROOT / "lightrag_storage"
DEFAULT_LLM_MODEL = os.getenv("LIGHTRAG_LLM_MODEL", "moonshotai/kimi-k3-free")
DEFAULT_EMBED_MODEL = os.getenv("LIGHTRAG_EMBED_MODEL", "openai/text-embedding-3-small")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.tokenrouter.com/v1")
EMBED_DIM = 1536
MIN_LLM_MAX_TOKENS = 2048


class CreditsExhaustedError(RuntimeError):
    """Raised when the LLM provider returns 402 (credits exhausted).

    Caught by the build loop to stop the pipeline cleanly so a new key can
    be supplied and the run resumed.
    """


CREDITS_FLAG_PATH = DEFAULT_STORAGE_DIR / "CREDITS_EXHAUSTED.flag"


class KeyPool:
    """Rotates through a list of API keys, skipping exhausted (402) ones.

    Shared by the LLM and embedding functions so a key marked dead by one is
    also skipped by the other. When every key is exhausted, `exhausted` is
    True and callers should stop the run.
    """

    def __init__(self, keys: List[str]):
        # Preserve order, drop duplicates/empties.
        self._keys = [k for i, k in enumerate(keys) if k and k not in keys[:i]]
        self._idx = 0
        self._dead: set = set()

    @property
    def current(self) -> str:
        if self._idx >= len(self._keys):
            raise CreditsExhaustedError(
                f"All {len(self._keys)} API key(s) exhausted — no live key available"
            )
        return self._keys[self._idx]

    @property
    def exhausted(self) -> bool:
        return len(self._dead) >= len(self._keys)

    def mark_dead(self, key: str) -> bool:
        """Mark `key` exhausted and advance. Returns True if a live key remains."""
        self._dead.add(key)
        while self._idx < len(self._keys) and self._keys[self._idx] in self._dead:
            self._idx += 1
        return not self.exhausted

    def __len__(self) -> int:
        return len(self._keys)


def load_openrouter_keys() -> Tuple[List[str], str]:
    """Load all OpenRouter keys: OPENROUTER_API_KEY plus any
    OPENROUTER_API_KEY_2, _3, ... (also accepts comma-separated list)."""
    load_dotenv()
    keys: List[str] = []
    primary = os.getenv("OPENROUTER_API_KEY", "")
    # Allow comma-separated keys in the primary var too.
    keys.extend(k.strip() for k in primary.split(",") if k.strip())
    for i in range(2, 20):
        extra = os.getenv(f"OPENROUTER_API_KEY_{i}", "").strip()
        if extra:
            keys.append(extra)
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    if not keys:
        raise ValueError("Set OPENROUTER_API_KEY in .env")
    return keys, base_url


def load_eval_key() -> Optional[str]:
    """Dedicated key for evaluation queries only (never used for indexing).

    Set EVAL_API_KEY in .env to keep eval traffic off the indexing keys.
    Returns None if not set (caller falls back to the indexing pool).
    """
    load_dotenv()
    key = os.getenv("EVAL_API_KEY", "").strip()
    return key or None


def _mark_credits_exhausted(detail: str) -> None:
    """Write a sentinel file so the build loop stops after the current batch.

    LightRAG swallows per-chunk exceptions (marks docs failed and continues),
    so raising alone is not enough to stop the run — the flag file is the
    reliable stop signal.
    """
    try:
        DEFAULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        CREDITS_FLAG_PATH.write_text(detail, encoding="utf-8")
    except OSError:
        pass

ENCYCLOPEDIA_FIELDS = [
    "entry_type",
    "icd_category",
    "body_systems",
    "severity_level",
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

MEDICAL_ENTITY_GUIDANCE = (
    "- Disease: medical diseases, disorders, and conditions\n"
    "- Symptom: symptoms, signs, and complaints\n"
    "- Treatment: drugs, therapies, and procedures\n"
    "- BodySystem: anatomical or organ systems\n"
    "- DiagnosticTest: tests and examinations"
)

EVAL_USER_PROMPT = (
    "You are a helpful medical information assistant. "
    "Answer using only the retrieved context. "
    "Keep your response concise (under 150 words). "
    "If the context is insufficient, say so briefly. "
    "End with: 'This is educational information, not a diagnosis.'"
)

ENTRY_LINE_RE = re.compile(r"^Entry:\s*(.+)$", re.MULTILINE)


def norm_entry_name(name: str) -> str:
    s = name.lower().strip()
    return re.sub(r"[\u2018\u2019\u02bc']", "", s)


def load_encyclopedia_name_index(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> Tuple[List[str], Dict[str, str]]:
    """Return canonical entry names and a normalized lookup map."""
    csv_path = dataset_dir / "01_medical_encyclopedia_entries.csv"
    entries = pd.read_csv(csv_path)
    names: List[str] = []
    norm_to_name: Dict[str, str] = {}
    for raw in entries["entry_name"].astype(str):
        name = raw.strip()
        if not name or name.lower().startswith("g al e"):
            continue
        names.append(name)
        norm_to_name[norm_entry_name(name)] = name
    return names, norm_to_name


def resolve_encyclopedia_entry(name: str, norm_to_name: Dict[str, str]) -> Optional[str]:
    candidate = name.strip()
    if not candidate:
        return None
    direct = norm_to_name.get(norm_entry_name(candidate))
    if direct:
        return direct

    cand_norm = norm_entry_name(candidate)
    for entry_norm, canonical in norm_to_name.items():
        if cand_norm == entry_norm:
            return canonical
        if len(cand_norm) >= 4 and (cand_norm in entry_norm or entry_norm in cand_norm):
            return canonical
    return None


def extract_ranked_entries_from_retrieval(
    retrieval_data: Optional[Dict[str, Any]],
    norm_to_name: Dict[str, str],
    max_items: int = 15,
) -> List[str]:
    """Build a ranked disease/entry list from LightRAG retrieval output."""
    if not retrieval_data:
        return []

    ranked: List[str] = []
    seen: set[str] = set()

    def add_candidate(raw_name: str) -> None:
        resolved = resolve_encyclopedia_entry(raw_name, norm_to_name)
        if resolved and resolved not in seen:
            seen.add(resolved)
            ranked.append(resolved)

    for chunk in retrieval_data.get("chunks") or []:
        content = str(chunk.get("content") or "")
        match = ENTRY_LINE_RE.search(content)
        if match:
            add_candidate(match.group(1).strip())

    for entity in retrieval_data.get("entities") or []:
        entity_name = str(entity.get("entity_name") or "").strip()
        entity_type = str(entity.get("entity_type") or "").lower()
        if not entity_name:
            continue
        if any(token in entity_type for token in ("disease", "disorder", "condition", "syndrome")):
            add_candidate(entity_name)
        elif resolve_encyclopedia_entry(entity_name, norm_to_name):
            add_candidate(entity_name)

    return ranked[:max_items]


def load_openrouter_env() -> Tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY in .env")
    return api_key, base_url


def load_kimi_env() -> Tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("KIMI_API_KEY", "").strip()
    base_url = os.getenv("KIMI_BASE_URL", "https://api.tokenrouter.com/v1").rstrip("/")
    if not api_key:
        raise ValueError("Set KIMI_API_KEY in .env")
    return api_key, base_url


def load_gemini_env() -> Tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    base_url = os.getenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
    ).rstrip("/")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in .env")
    return api_key, base_url


def openrouter_client_configs() -> dict:
    headers = {"Content-Type": "application/json"}
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.getenv("OPENROUTER_X_TITLE", "MediBot-LightRAG").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return {"default_headers": headers}


def _extract_openrouter_message_content(message: Dict[str, Any]) -> str:
    """OpenRouter reasoning models may leave content empty and fill reasoning instead."""
    content = (message.get("content") or "").strip()
    if content:
        return content
    for key in ("reasoning", "reasoning_content"):
        alt = (message.get(key) or "").strip()
        if alt:
            return alt
    return ""


def make_openrouter_llm(api_key, base_url: str, model: str = DEFAULT_LLM_MODEL):
    # Accept either a single key string or a shared KeyPool for rotation.
    pool = api_key if isinstance(api_key, KeyPool) else KeyPool([api_key])
    url = f"{base_url.rstrip('/')}/chat/completions"
    extra_headers = openrouter_client_configs().get("default_headers", {})

    async def llm_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        enable_cot: bool = False,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        del enable_cot, keyword_extraction
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": str(prompt)})

        max_tokens = max(int(kwargs.pop("max_tokens", MIN_LLM_MAX_TOKENS)), MIN_LLM_MAX_TOKENS)
        temperature = float(kwargs.pop("temperature", 0.25))
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        timeout = httpx.Timeout(360.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            while True:
                key = pool.current
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                headers.update(extra_headers)
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 402:
                    if pool.mark_dead(key):
                        print(f"[KeyPool] Key ...{key[-6:]} exhausted (402); "
                              f"switching to ...{pool.current[-6:]} ({len(pool._dead)}/{len(pool)} dead)")
                        continue  # retry with next key
                    detail = f"All {len(pool)} LLM API key(s) exhausted (HTTP 402): {resp.text[:300]}"
                    _mark_credits_exhausted(detail)
                    raise CreditsExhaustedError(detail)
                if resp.status_code >= 400:
                    raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:400]}")
                data = resp.json()
                break

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as ex:
            raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from ex

        content = _extract_openrouter_message_content(message)
        if not content:
            raise RuntimeError("OpenRouter returned empty content and reasoning fields")
        return content

    return llm_func


def make_openrouter_embed(api_key, base_url: str, model: str = DEFAULT_EMBED_MODEL):
    # Accept either a single key string or a shared KeyPool for rotation.
    pool = api_key if isinstance(api_key, KeyPool) else KeyPool([api_key])

    def _build(key: str):
        return partial(
            openai_embed,
            model=model,
            api_key=key,
            base_url=base_url,
            client_configs=openrouter_client_configs(),
        )

    async def embed_with_credit_guard(texts):
        while True:
            key = pool.current
            try:
                return await _build(key)(texts)
            except Exception as ex:
                msg = str(ex)
                if "402" in msg and ("credit" in msg.lower() or "insufficient" in msg.lower()):
                    if pool.mark_dead(key):
                        print(f"[KeyPool] Embed key ...{key[-6:]} exhausted (402); "
                              f"switching to ...{pool.current[-6:]} ({len(pool._dead)}/{len(pool)} dead)")
                        continue  # retry with next key
                    detail = f"All {len(pool)} embedding API key(s) exhausted (HTTP 402): {ex}"
                    _mark_credits_exhausted(detail)
                    raise CreditsExhaustedError(detail) from ex
                raise

    return EmbeddingFunc(embedding_dim=EMBED_DIM, func=embed_with_credit_guard)


def clean_field(value: object, max_len: int = 900) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def entry_to_document(row: pd.Series, max_field: int = 900) -> str:
    name = clean_field(row.get("entry_name", ""), 120)
    if not name:
        return ""
    parts = [f"Entry: {name}"]
    for field in ENCYCLOPEDIA_FIELDS:
        value = clean_field(row.get(field, ""), max_field)
        if value:
            parts.append(f"{field.replace('_', ' ').title()}: {value}")
    return "\n".join(parts)


def load_corpus_documents(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    limit: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    csv_path = dataset_dir / "01_medical_encyclopedia_entries.csv"
    entries = pd.read_csv(csv_path)
    docs: List[str] = []
    ids: List[str] = []
    names: List[str] = []

    for idx, row in entries.iterrows():
        name = clean_field(row.get("entry_name", ""), 120)
        if not name or name.lower().startswith("g al e"):
            continue
        doc = entry_to_document(row)
        if len(doc.split()) < 20:
            continue
        doc_id = f"enc_{idx:05d}"
        docs.append(doc)
        ids.append(doc_id)
        names.append(name)
        if limit is not None and len(docs) >= limit:
            break

    return docs, ids, names


def storage_is_ready(storage_dir: Path) -> bool:
    if not storage_dir.exists():
        return False
    markers = [
        storage_dir / "kv_store_full_docs.json",
        storage_dir / "vdb_chunks.json",
    ]
    return any(p.exists() for p in markers)


def reset_storage(storage_dir: Path) -> None:
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)


def strip_lightrag_response(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


async def create_lightrag(
    storage_dir: Path = DEFAULT_STORAGE_DIR,
    llm_model: str = DEFAULT_LLM_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    use_eval_key: bool = False,
) -> LightRAG:
    openrouter_keys, openrouter_base_url = load_openrouter_keys()

    # For evaluation, prefer the dedicated EVAL_API_KEY so query traffic never
    # touches the indexing keys.
    if use_eval_key:
        eval_key = load_eval_key()
        if eval_key:
            pool = KeyPool([eval_key])
            print("[KeyPool] Using dedicated EVAL_API_KEY for queries")
        else:
            pool = KeyPool(openrouter_keys)
            print("[KeyPool] EVAL_API_KEY not set; falling back to indexing keys")
    else:
        pool = KeyPool(openrouter_keys)
        print(f"[KeyPool] Loaded {len(pool)} OpenRouter API key(s)")
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Route LLM calls: Kimi models use the Kimi endpoint, Gemini models use the
    # Gemini OpenAI-compatible endpoint, everything else uses OpenRouter
    if llm_model.startswith("moonshotai/kimi"):
        llm_key, llm_base_url = load_kimi_env()
        llm_func = make_openrouter_llm(llm_key, llm_base_url, llm_model)
    elif llm_model.startswith("gemini"):
        llm_key, llm_base_url = load_gemini_env()
        llm_func = make_openrouter_llm(llm_key, llm_base_url, llm_model)
    else:
        llm_func = make_openrouter_llm(pool, openrouter_base_url, llm_model)

    rag = LightRAG(
        working_dir=str(storage_dir),
        llm_model_func=llm_func,
        embedding_func=make_openrouter_embed(pool, openrouter_base_url, embed_model),
        addon_params={
            "language": "English",
            "entity_types_guidance": MEDICAL_ENTITY_GUIDANCE,
            "chunker": {
                "chunk_token_size": 1000,
                "chunk_overlap_token_size": 80,
            },
        },
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def query_lightrag(
    rag: LightRAG,
    query: str,
    mode: str = "mix",
    user_prompt: str = EVAL_USER_PROMPT,
) -> str:
    full = await query_lightrag_full(rag, query, mode=mode, user_prompt=user_prompt)
    return full["response"]


async def query_lightrag_full(
    rag: LightRAG,
    query: str,
    mode: str = "mix",
    user_prompt: str = EVAL_USER_PROMPT,
    norm_to_name: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run LightRAG once and return LLM text plus ranked encyclopedia entries."""
    result = await rag.aquery_llm(
        query,
        param=QueryParam(
            mode=mode,
            user_prompt=user_prompt,
            response_type="Single Paragraph",
            enable_rerank=False,
        ),
    )
    retrieval_data = result.get("data") if isinstance(result.get("data"), dict) else {}
    llm_payload = result.get("llm_response") or {}
    response = strip_lightrag_response(str(llm_payload.get("content") or ""))

    ranked: List[str] = []
    if norm_to_name:
        ranked = extract_ranked_entries_from_retrieval(retrieval_data, norm_to_name)

    return {
        "response": response,
        "ranked": ranked,
        "retrieval_data": retrieval_data,
        "status": result.get("status", ""),
        "message": result.get("message", ""),
    }
