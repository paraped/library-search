"""Configuration management — persisted in ~/.config/library-search/config.json"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_FILE = Path.home() / ".config" / "library-search" / "config.json"

DEFAULTS: dict = {
    "books_dir": str(Path.home() / "Book_Library"),
    "index_dir": str(Path.home() / ".local" / "share" / "library-search" / "index"),
    "port": 8765,
    "llm_api_key": "",
    "llm_model": "claude-haiku-4-5-20251001",
    "chunk_size_tokens": 200,     # well under all-MiniLM-L6-v2's 256-token limit
    "chunk_overlap_tokens": 20,
    "batch_size": 64,
    "topics_max_chars": 2500,
}


def load() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULTS, **json.load(f)}
    return dict(DEFAULTS)


def save(updates: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    current = load()
    current.update(updates)
    with open(CONFIG_FILE, "w") as f:
        json.dump(current, f, indent=2)


def books_dir() -> Path:
    p = Path(load()["books_dir"]).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def index_dir() -> Path:
    p = Path(load()["index_dir"]).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def port() -> int:
    return int(load().get("port", 8765))
