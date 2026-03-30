"""Standalone HTTP server for library search — no MCP required."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

HTML_FILE = Path(__file__).resolve().parent / "library.html"

app = FastAPI(title="Library Search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


class SearchRequest(BaseModel):
    query: str
    topic: str = ""
    n: int = 8


class SettingsUpdate(BaseModel):
    books_dir: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    port: int = 0


@app.get("/")
def serve_ui():
    if not HTML_FILE.exists():
        raise HTTPException(status_code=404, detail="library.html not found")
    return FileResponse(HTML_FILE)


@app.get("/books")
def list_books():
    from indexer import list_indexed_books
    return list_indexed_books(cfg.index_dir())


@app.post("/search")
def search(req: SearchRequest):
    from indexer import search as _search
    hits, _ = _search(
        req.query,
        cfg.index_dir(),
        get_embedder(),
        n=req.n,
        topic=req.topic or None,
    )
    return hits


@app.post("/tag")
def tag_books():
    from indexer import tag_all_books
    return tag_all_books(cfg.books_dir(), cfg.index_dir())


@app.get("/settings")
def get_settings():
    c = cfg.load()
    return {
        "books_dir": c["books_dir"],
        "index_dir": c["index_dir"],
        "port": c["port"],
        "llm_api_key_set": bool(c.get("llm_api_key", "").strip()),
        "llm_model": c.get("llm_model", "claude-haiku-4-5-20251001"),
    }


@app.post("/settings")
def update_settings(data: SettingsUpdate):
    updates: dict = {}
    if data.books_dir:
        updates["books_dir"] = data.books_dir
    if data.llm_api_key:
        updates["llm_api_key"] = data.llm_api_key
    if data.llm_model:
        updates["llm_model"] = data.llm_model
    if data.port:
        updates["port"] = data.port
    if updates:
        cfg.save(updates)
    return {"ok": True, "restart_required": bool(data.books_dir or data.port)}


if __name__ == "__main__":
    import uvicorn
    p = cfg.port()
    print(f"Library search running at http://localhost:{p}")
    uvicorn.run(app, host="127.0.0.1", port=p)
