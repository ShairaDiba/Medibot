#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
MEDIBOT = ROOT / "medibot.py"


def _parse_output(stdout: str) -> Dict[str, object]:
    lines = stdout.splitlines()

    def section(name: str) -> List[str]:
        out: List[str] = []
        in_sec = False
        marker = f"=== {name} ==="
        for ln in lines:
            if ln.strip() == marker:
                in_sec = True
                continue
            if in_sec and ln.startswith("=== ") and ln.strip().endswith(" ==="):
                break
            if in_sec:
                out.append(ln)
        return out

    seed_block = section("Graph seeds (node_ids)")
    seed_graph_nodes_block = section("Seed graph nodes")
    ranked_block = section("Ranked diseases (graph-traversal score)")
    trace_block = section("Seed trace (sample)")

    seeds: List[str] = []
    if seed_block:
        seed_line = " ".join(s.strip() for s in seed_block if s.strip())
        if seed_line and not seed_line.startswith("(none"):
            seeds = [x.strip() for x in seed_line.split(",") if x.strip()]
    elif seed_graph_nodes_block:
        for ln in seed_graph_nodes_block:
            v = ln.strip()
            if v:
                seeds.append(v)

    ranked: List[Dict[str, object]] = []
    for ln in ranked_block:
        m = re.match(r"^(.*?)\s+([0-9]+(?:\.[0-9]+)?)\s*$", ln.rstrip())
        if not m:
            continue
        ranked.append({"name": m.group(1).strip(), "score": float(m.group(2))})

    trace = [ln.strip() for ln in trace_block if ln.strip()]

    llm_response = ""
    for hdr in ("=== OpenRouter response ===", "=== Groq response ===", "=== Gemini response ==="):
        pos = stdout.find(hdr)
        if pos != -1:
            llm_response = stdout[pos + len(hdr):].strip()
            break

    backend = "csv"
    m_backend = re.search(r"=== Graph backend ===\s*\n([^\n]+)", stdout)
    if m_backend:
        backend = m_backend.group(1).strip()

    visited_summary = ""
    m_vs = re.search(r"Visited region summary:\s*(.*)", stdout)
    if m_vs:
        visited_summary = m_vs.group(1).strip()

    definition_preview = ""
    m_def = re.search(
        r"--- Encyclopedia \(preview\) ---\s*(.*?)\s*=== Seed graph nodes ===",
        stdout,
        re.DOTALL,
    )
    if m_def:
        definition_preview = m_def.group(1).strip()

    return {
        "seeds": seeds,
        "ranked": ranked,
        "trace": trace,
        "backend_used": backend,
        "visited_summary": visited_summary,
        "definition_preview": definition_preview,
        "response": llm_response,
        "raw": stdout,
    }


class Handler(SimpleHTTPRequestHandler):
    def _json(self, code: int, payload: Dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path == "/api/translate":
            self._handle_translate()
            return
        if self.path != "/api/query":
            self._json(404, {"ok": False, "error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception as ex:
            self._json(400, {"ok": False, "error": f"Invalid JSON: {ex}"})
            return

        query = str(data.get("query", "")).strip()
        if not query:
            self._json(400, {"ok": False, "error": "query is required"})
            return

        backend = str(data.get("backend", "csv")).strip().lower()
        if backend not in ("csv", "neo4j"):
            backend = "csv"
        model = str(data.get("model", "")).strip() or "openai/gpt-oss-120b"
        no_llm = bool(data.get("no_llm", False))

        env = os.environ.copy()
        api_key = str(data.get("api_key", "")).strip()
        if api_key:
            env["OPENROUTER_API_KEY"] = api_key

        neo4j = data.get("neo4j", {}) if isinstance(data.get("neo4j", {}), dict) else {}
        for key, env_name in (
            ("uri", "NEO4J_URI"),
            ("user", "NEO4J_USER"),
            ("password", "NEO4J_PASSWORD"),
            ("database", "NEO4J_DATABASE"),
        ):
            v = str(neo4j.get(key, "")).strip()
            if v:
                env[env_name] = v

        cmd = [
            sys.executable,
            str(MEDIBOT),
            "--dataset-dir",
            "final_dataset",
            "--user-input",
            query,
            "--backend",
            backend,
            "--model",
            model,
        ]
        if no_llm:
            cmd.append("--no-llm")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=240,
            )
        except subprocess.TimeoutExpired:
            self._json(504, {"ok": False, "error": "medibot timed out"})
            return

        stdout = proc.stdout or ""
        stderr = (proc.stderr or "").strip()

        if proc.returncode != 0:
            self._json(
                500,
                {
                    "ok": False,
                    "error": stderr or "medibot failed",
                    "stdout": stdout,
                    "returncode": proc.returncode,
                },
            )
            return

        parsed = _parse_output(stdout)
        self._json(200, {"ok": True, **parsed})

    def _handle_translate(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception as ex:
            self._json(400, {"ok": False, "error": f"Invalid JSON: {ex}"})
            return

        source_text = str(data.get("text", "")).strip()
        if not source_text:
            self._json(400, {"ok": False, "error": "text is required"})
            return

        api_key = str(data.get("api_key", "")).strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            self._json(400, {"ok": False, "error": "OpenRouter API key missing"})
            return

        model = str(data.get("model", "")).strip() or "openai/gpt-oss-120b"
        prompt = (
            "Translate the following medical assistant response into natural Bangla. "
            "Keep the meaning exactly the same, do not add new facts, and preserve line breaks where possible. "
            "Return only Bangla text.\n\n"
            f"{source_text}"
        )

        try:
            translated = _query_openrouter(api_key=api_key, model=model, prompt=prompt)
        except Exception as ex:
            self._json(500, {"ok": False, "error": str(ex)})
            return

        self._json(200, {"ok": True, "translated_text": translated})


def _query_openrouter(api_key: str, model: str, prompt: str) -> str:
    base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 3000,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload, headers=headers)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as ex:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from ex


def main() -> None:
    load_dotenv()
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", 8787), Handler)
    print("MediBot UI server running: http://127.0.0.1:8787/UI.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
