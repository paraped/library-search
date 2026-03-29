"""Library search MCP server — semantic search over PDF/epub collection."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from watchfiles import Change, watch

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

BOOKS_DIR = cfg.books_dir()
INDEX_DIR = cfg.index_dir()

mcp = FastMCP("library-search")

# Lazy-loaded embedder — loaded on first use to keep startup fast
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _auto_tag(filename: str) -> None:
    """Try LLM topic tagging on a single newly-indexed book. Silently ignores errors."""
    try:
        from indexer import get_collection, list_indexed_books, update_book_topics
        from topics import detect_topics_llm, get_book_intro_text
        collection = get_collection(INDEX_DIR)
        books = {b["file"]: b for b in list_indexed_books(INDEX_DIR)}
        book = books.get(filename)
        if not book:
            return
        path = BOOKS_DIR / filename
        if not path.exists():
            return
        intro = get_book_intro_text(path)
        topics = detect_topics_llm(book["title"], book["author"], intro)
        if topics:
            update_book_topics(filename, topics, collection)
            print(f"[library-search] tagged: {filename} → {topics}", flush=True)
    except Exception as e:
        print(f"[library-search] auto-tag skipped for {filename}: {e}", flush=True)


def _index_untracked() -> None:
    """Index any files in BOOKS_DIR that are not yet in the index."""
    from indexer import get_collection, index_file, list_indexed_books
    collection = get_collection(INDEX_DIR)
    indexed = {b["file"] for b in list_indexed_books(INDEX_DIR)}
    for p in sorted(BOOKS_DIR.iterdir()):
        if p.suffix.lower() in {".pdf", ".epub"} and p.name not in indexed:
            try:
                meta = index_file(p, collection, _get_embedder())
                print(
                    f"[library-search] startup-indexed: {meta['title']}"
                    f" ({meta['chunks']} chunks)",
                    flush=True,
                )
                _auto_tag(p.name)
            except Exception as e:
                print(f"[library-search] startup-index failed for {p.name}: {e}", flush=True)


def _watch_books() -> None:
    """Background daemon: index untracked files on startup, then watch for new drops."""
    _index_untracked()
    for changes in watch(str(BOOKS_DIR)):
        for change_type, path_str in changes:
            if change_type == Change.added:
                p = Path(path_str)
                if p.suffix.lower() in {".pdf", ".epub"}:
                    try:
                        from indexer import get_collection, index_file
                        collection = get_collection(INDEX_DIR)
                        meta = index_file(p, collection, _get_embedder())
                        print(
                            f"[library-search] auto-indexed: {meta['title']}"
                            f" ({meta['chunks']} chunks)",
                            flush=True,
                        )
                        _auto_tag(p.name)
                    except Exception as e:
                        print(f"[library-search] auto-index failed for {p.name}: {e}", flush=True)


threading.Thread(target=_watch_books, daemon=True, name="books-watcher").start()


@mcp.tool()
def search_books(query: str, n: int = 5, topic: str = "") -> str:
    """Semantic search across all indexed books.

    Args:
        query: Natural language query — topic, concept, or question.
        n: Number of results to return (default 5).
        topic: Optional topic filter (e.g. 'building', 'conservation'). Only searches
               books tagged with this topic. Leave empty to search all books.

    Returns:
        JSON list of matching passages with title, author, page, topics, and relevance score.
    """
    from indexer import search
    hits = search(query, INDEX_DIR, _get_embedder(), n=n, topic=topic or None)
    if not hits:
        return "No results found. Make sure books are indexed with index_library first."
    return json.dumps(hits, ensure_ascii=False, indent=2)


@mcp.tool()
def list_books() -> str:
    """List all books currently indexed in the library.

    Returns:
        JSON list of books with title, author, filename, and topics.
    """
    from indexer import list_indexed_books
    books = list_indexed_books(INDEX_DIR)
    if not books:
        return "No books indexed yet. Use index_library to index books from Work/library/books/."
    return json.dumps(books, ensure_ascii=False, indent=2)


@mcp.tool()
def index_book(path: str) -> str:
    """Index a single PDF or epub file into the library.

    Args:
        path: Absolute path to the file, or filename if it's in Work/library/books/.

    Returns:
        Confirmation with title, author, page count, and chunk count.
    """
    from indexer import get_collection, index_file
    p = Path(path)
    if not p.is_absolute():
        p = BOOKS_DIR / p
    if not p.exists():
        return f"File not found: {p}"
    collection = get_collection(INDEX_DIR)
    meta = index_file(p, collection, _get_embedder())
    _auto_tag(p.name)
    return f"Indexed: {meta['title']} by {meta['author']} — {meta['pages']} pages, {meta['chunks']} chunks"


@mcp.tool()
def index_library() -> str:
    """Batch-index all PDF and epub files in Work/library/books/.

    Returns:
        Summary of indexed files with status for each.
    """
    from indexer import index_library as _index_library
    results = _index_library(BOOKS_DIR, INDEX_DIR, _get_embedder())
    if not results:
        return f"No supported files found in {BOOKS_DIR}. Drop PDFs or epubs there first."
    lines = []
    for r in results:
        if r["status"] == "ok":
            lines.append(f"✓ {r['title']} ({r['file']}) — {r['pages']} pages, {r['chunks']} chunks")
        else:
            lines.append(f"✗ {r['file']}: {r['error']}")
    return "\n".join(lines)


@mcp.tool()
def tag_books() -> str:
    """Detect and assign topic tags to all indexed books.

    Runs two strategies and shows a comparison:
    - LLM (Claude Haiku): infers topics from title + first pages. Requires connectivity.
    - Clustering: groups books by embedding similarity, labels with TF-IDF keywords. Always works offline.

    LLM result is used when available; clustering is the fallback.
    Updates topics in the index so search_books(topic=...) filtering works.

    Returns:
        Comparison table showing both strategy results and the stored final topics.
    """
    from indexer import tag_all_books
    results = tag_all_books(BOOKS_DIR, INDEX_DIR)
    if not results:
        return "No books indexed yet."
    lines = []
    for r in results:
        lines.append(f"\n{r['title']} ({r['file']})")
        if r["llm_error"]:
            lines.append(f"  LLM:        [error: {r['llm_error'][:80]}]")
        else:
            lines.append(f"  LLM:        {', '.join(r['llm_topics']) or '(none)'}")
        if r["cluster_error"]:
            lines.append(f"  Clustering: [error: {r['cluster_error'][:80]}]")
        else:
            lines.append(f"  Clustering: {', '.join(r['cluster_topics']) or '(none)'}")
        lines.append(f"  → Stored:   {', '.join(r['final_topics']) or '(none)'}  [{r['source']}]")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
