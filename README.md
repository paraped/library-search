# library-search

Semantic search over a personal PDF and EPUB collection. Drop books in a folder — they are indexed automatically and searchable instantly via a local web app or from Claude conversations.

## Features

- **Semantic search** — finds relevant passages by meaning, not just keywords
- **Auto-indexing** — drop a PDF into the books folder and it's indexed within seconds
- **Topic detection** — books are automatically tagged (LLM via Anthropic API, or offline clustering fallback)
- **Dual interface** — standalone web app (Homebrew) and Claude MCP integration
- **Fully offline** — all indexing and search runs locally; LLM tagging is optional

---

## Installation

### Option A — Homebrew (standalone web app)

```bash
brew tap paraped/library-search
brew install library-search
brew services start library-search
```

Open **http://localhost:8765** in your browser.

Drop PDFs into `~/Book_Library/` — they are indexed automatically.

### Option B — Claude MCP integration

```bash
git clone https://github.com/paraped/library-search ~/.claude/mcp-servers/library-search
claude mcp add -s user library-search -- \
  uv run --project ~/.claude/mcp-servers/library-search \
  python ~/.claude/mcp-servers/library-search/server.py
```

---

## Usage

### Web UI

| Tab | What it does |
|---|---|
| **Search** | Enter a query; optionally filter by topic and number of results |
| **Books** | List all indexed books with authors and topic tags |
| **Settings** | Set books directory, Anthropic API key, LLM model |

### MCP tools (Claude integration)

| Tool | Description |
|---|---|
| `search_books(query, n, topic)` | Semantic search; `topic` filters to matching books only |
| `list_books` | List all indexed books with titles, authors, and topics |
| `index_book(path)` | Index a single PDF or EPUB |
| `index_library` | Batch-index all books in the books folder |
| `tag_books` | Detect and assign topic tags; shows LLM vs clustering comparison |

---

## Configuration

Config file: `~/.config/library-search/config.json`

| Key | Default | Description |
|---|---|---|
| `books_dir` | `~/Book_Library` | Folder to watch for PDFs/EPUBs |
| `index_dir` | `~/.local/share/library-search/index` | ChromaDB vector store location |
| `port` | `8765` | Web server port |
| `llm_api_key` | `""` | Anthropic API key (optional — for LLM topic tagging) |
| `llm_model` | `claude-haiku-4-5-20251001` | Model used for topic tagging |

Settings can also be changed in the web UI **Settings** tab. Path and port changes require a server restart.

---

## Topic detection

When a new book is indexed, topics are assigned automatically:

1. **LLM** (primary) — Claude Haiku reads the title, author, and first 3 pages, then assigns 1–3 tags like `building`, `conservation`, `history`. Requires an Anthropic API key in settings.
2. **Clustering** (offline fallback) — books are grouped by embedding similarity; clusters are labelled with TF-IDF keywords. Works without any API key.

Run `tag_books` to re-tag all books and see a side-by-side comparison of both strategies.

---

## How it works

```
PDFs / EPUBs
     │
     │  PyMuPDF — text extraction per page
     ▼
 text chunks (~500 words, 50-word overlap)
     │
     │  sentence-transformers (all-MiniLM-L6-v2, ~80 MB, runs on CPU)
     ▼
 384-dimension embeddings
     │
     │  ChromaDB — persistent local vector store (cosine similarity)
     ▼
 index on disk

Search query → embedding → cosine similarity → top N chunks with metadata
```

Title and author are extracted from PDF metadata, falling back to font-size heuristics on the first 8 pages.

---

## Stack

| Library | Role |
|---|---|
| [FastMCP](https://github.com/jlowin/fastmcp) | MCP server framework |
| [FastAPI](https://fastapi.tiangolo.com) | Web server |
| [ChromaDB](https://www.trychroma.com) | Local vector database |
| [sentence-transformers](https://sbert.net) | Embedding model |
| [PyMuPDF](https://pymupdf.readthedocs.io) | PDF/EPUB text extraction |
| [scikit-learn](https://scikit-learn.org) | Clustering + TF-IDF for topic detection |
| [watchfiles](https://watchfiles.helpmanual.io) | File system watcher for auto-indexing |
| [uv](https://github.com/astral-sh/uv) | Python package manager |
