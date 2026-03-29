"""PDF/epub extraction, chunking, and ChromaDB ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chromadb
import fitz  # PyMuPDF

SUPPORTED_EXTENSIONS = {".pdf", ".epub"}
CHUNK_SIZE = 500  # approximate tokens (words)
CHUNK_OVERLAP = 50


def _book_id(path: Path) -> str:
    return hashlib.md5(path.name.encode()).hexdigest()[:12]


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text) for each page."""
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append((i, text))
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def _looks_like_name(text: str) -> bool:
    """Heuristic: short, title-cased (not ALL CAPS), no noise tokens."""
    _NOISE = {'©', 'http', 'www', 'isbn', 'copyright', 'published', 'edition', 'press', 'inc.', 'llc', 'ltd'}
    if not (3 < len(text) < 80):
        return False
    lower = text.lower()
    if any(n in lower for n in _NOISE):
        return False
    # Reject all-caps strings (subtitles / chapter headers)
    if text == text.upper() and any(c.isalpha() for c in text):
        return False
    return True


def _extract_from_first_pages(doc) -> tuple[str, str]:
    """Try to infer title and author from the first 8 pages using font-size heuristics."""
    # Collect spans from all candidate pages first
    pages_spans: list[list[tuple[float, str]]] = []
    for page_num in range(min(8, len(doc))):
        page = doc[page_num]
        spans = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    if text and len(text) > 2:
                        spans.append((size, text))
        pages_spans.append(spans)

    # Find the title page: whichever page has the single largest font
    title_page_idx = -1
    global_max_size = 0.0
    for i, spans in enumerate(pages_spans):
        if spans:
            page_max = max(s[0] for s in spans)
            if page_max > global_max_size:
                global_max_size = page_max
                title_page_idx = i

    title = ""
    author = ""

    if title_page_idx == -1:
        return title, author

    # Extract title: largest font spans on the title page
    spans = sorted(pages_spans[title_page_idx], key=lambda x: -x[0])
    max_size = spans[0][0]
    title_parts = [s[1] for s in spans if s[0] >= max_size * 0.85][:4]
    title = " ".join(title_parts)

    # Search for author across all pages, starting from the title page
    search_order = list(range(title_page_idx, len(pages_spans))) + list(range(0, title_page_idx))
    for idx in search_order:
        spans = sorted(pages_spans[idx], key=lambda x: -x[0])
        texts = [s[1] for s in spans]

        # Explicit "by X" / "Author: X" patterns
        for i, text in enumerate(texts):
            lower = text.lower()
            if lower == "by" and i + 1 < len(texts):
                candidate = texts[i + 1].strip()
                if _looks_like_name(candidate):
                    author = candidate
                    break
            if lower.startswith("by ") and len(text) > 4:
                author = text[3:].strip()
                break
            if lower.startswith("author:"):
                author = text.split(":", 1)[-1].strip()
                break
        if author:
            break

        # Fallback on title page: among second-largest font spans, find one that
        # looks like a person's name (title-cased, short, not ALL CAPS subtitle)
        if idx == title_page_idx and len(spans) > 1:
            page_max = spans[0][0]
            # Collect all distinct sizes below the title size
            sub_sizes = sorted({s[0] for s in spans if s[0] < page_max * 0.85}, reverse=True)
            for size in sub_sizes[:3]:
                candidates = [s[1] for s in spans if abs(s[0] - size) < 1]
                for candidate in candidates:
                    if _looks_like_name(candidate):
                        author = candidate
                        break
                if author:
                    break

        if author:
            break

    return title, author


def guess_metadata(path: Path) -> dict[str, str]:
    """Extract title/author from PDF metadata, then first-page analysis, fall back to filename."""
    doc = fitz.open(str(path))
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip()
    author = (meta.get("author") or "").strip()
    if not title or not author:
        inferred_title, inferred_author = _extract_from_first_pages(doc)
        if not title:
            title = inferred_title
        if not author:
            author = inferred_author
    doc.close()
    if not title:
        title = path.stem.replace("_", " ").replace("-", " ").title()
    if not author:
        author = "Unknown"
    return {"title": title, "author": author, "file": path.name}


def get_collection(index_dir: Path) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(index_dir))
    return client.get_or_create_collection(
        name="library",
        metadata={"hnsw:space": "cosine"},
    )


def index_file(path: Path, collection: chromadb.Collection, embedder) -> dict:
    """Index a single file. Returns metadata dict."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {path.suffix}")

    meta = guess_metadata(path)
    book_id = _book_id(path)

    # Remove existing chunks for this book (re-index support)
    existing = collection.get(where={"file": path.name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    pages = extract_pages(path)
    all_chunks = []
    all_ids = []
    all_metas = []

    chunk_index = 0
    for page_num, page_text in pages:
        for chunk in chunk_text(page_text):
            chunk_id = f"{book_id}_{chunk_index}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metas.append({
                "title": meta["title"],
                "author": meta["author"],
                "file": meta["file"],
                "page": page_num,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    if not all_chunks:
        raise ValueError(f"No text extracted from {path.name}")

    # Embed in batches of 64
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i + batch_size]
        embeddings = embedder.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            ids=all_ids[i:i + batch_size],
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=all_metas[i:i + batch_size],
        )

    meta["chunks"] = chunk_index
    meta["pages"] = len(pages)
    return meta


def index_library(books_dir: Path, index_dir: Path, embedder) -> list[dict]:
    """Index all supported files in books_dir. Returns list of metadata dicts."""
    books_dir = Path(books_dir)
    collection = get_collection(index_dir)
    results = []
    for path in sorted(books_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                meta = index_file(path, collection, embedder)
                results.append({"status": "ok", **meta})
            except Exception as e:
                results.append({"status": "error", "file": path.name, "error": str(e)})
    return results


def update_book_topics(filename: str, topics: list[str], collection: chromadb.Collection) -> int:
    """Overwrite the 'topics' field on every chunk for a book. Returns chunk count."""
    existing = collection.get(where={"file": filename}, include=["metadatas"])
    if not existing["ids"]:
        return 0
    topics_str = ",".join(topics)
    new_metas = [{**m, "topics": topics_str} for m in existing["metadatas"]]
    collection.update(ids=existing["ids"], metadatas=new_metas)
    return len(existing["ids"])


def tag_all_books(books_dir: Path, index_dir: Path) -> list[dict]:
    """Run both LLM and clustering topic detection on all indexed books.

    Returns a comparison list. Stores the best available result per book:
    LLM topics when reachable, clustering topics as fallback.
    """
    from topics import detect_topics_clustering, detect_topics_llm, get_book_intro_text

    collection = get_collection(index_dir)
    books = list_indexed_books(index_dir)
    if not books:
        return []

    book_files = [b["file"] for b in books]
    book_map = {b["file"]: b for b in books}

    # Strategy 4: clustering (always runs, offline-safe)
    try:
        cluster_topics = detect_topics_clustering(book_files, collection, books_dir)
        cluster_error = None
    except Exception as e:
        cluster_topics = {f: [] for f in book_files}
        cluster_error = str(e)

    results = []
    for fname in book_files:
        book = book_map[fname]
        path = books_dir / fname

        # Strategy 3: LLM (requires connectivity)
        llm_topics: list[str] = []
        llm_error: str | None = None
        if path.exists():
            try:
                intro = get_book_intro_text(path)
                llm_topics = detect_topics_llm(book["title"], book["author"], intro)
            except Exception as e:
                llm_error = str(e)

        final_topics = llm_topics if llm_topics else cluster_topics.get(fname, [])
        update_book_topics(fname, final_topics, collection)

        results.append({
            "file": fname,
            "title": book["title"],
            "llm_topics": llm_topics,
            "llm_error": llm_error,
            "cluster_topics": cluster_topics.get(fname, []),
            "cluster_error": cluster_error,
            "final_topics": final_topics,
            "source": "llm" if llm_topics else ("clustering" if final_topics else "none"),
        })

    return results


def list_indexed_books(index_dir: Path) -> list[dict]:
    """Return deduplicated list of books in the index."""
    collection = get_collection(index_dir)
    all_items = collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in all_items["metadatas"]:
        file = meta["file"]
        if file not in seen:
            seen[file] = {
                "title": meta["title"],
                "author": meta["author"],
                "file": file,
                "topics": [t for t in meta.get("topics", "").split(",") if t],
            }
    return list(seen.values())


def search(query: str, index_dir: Path, embedder, n: int = 5, topic: str | None = None) -> list[dict]:
    """Semantic search. Returns top n chunks with metadata.

    If topic is given, restricts search to books tagged with that topic.
    """
    collection = get_collection(index_dir)
    total = collection.count()
    if total == 0:
        return []

    where = None
    if topic:
        all_items = collection.get(include=["metadatas"])
        matching_files: set[str] = set()
        for meta in all_items["metadatas"]:
            stored = meta.get("topics", "")
            if any(topic.lower() in t.lower().strip() for t in stored.split(",") if t):
                matching_files.add(meta["file"])
        if matching_files:
            where = {"file": {"$in": list(matching_files)}}
        # if nothing tagged, search everything (graceful degradation)

    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n, total),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
    except Exception:
        # where filter may fail if no docs match; retry without filter
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n, total),
            include=["documents", "metadatas", "distances"],
        )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "title": meta["title"],
            "author": meta["author"],
            "file": meta["file"],
            "page": meta["page"],
            "topics": [t for t in meta.get("topics", "").split(",") if t],
            "score": round(1 - dist, 3),
            "passage": doc,
        })
    return hits
