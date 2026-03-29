"""PDF/epub extraction, chunking, and ChromaDB ingestion."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import chromadb
import fitz  # PyMuPDF

SUPPORTED_EXTENSIONS = {".pdf", ".epub"}
INDEX_VERSION = "v2_token"

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return _tokenizer


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


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """Split text into overlapping token-based chunks.

    Uses the embedding model's tokenizer to avoid silent truncation at the
    model's 256-token context limit. Default chunk_size=200 leaves headroom
    for special tokens added during encoding.
    """
    tokenizer = _get_tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    chunks = []
    start = 0
    while start < len(ids):
        end = start + chunk_size
        chunks.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        if end >= len(ids):
            break
        start += chunk_size - overlap
    return chunks


def parse_topics(raw: str) -> list[str]:
    """Parse topics from stored metadata. Handles JSON array and legacy comma-string."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(t) for t in parsed if str(t).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return [t.strip() for t in raw.split(",") if t.strip()]


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


def get_index_version(index_dir: Path) -> str:
    """Read the index schema version. Returns 'v1_word' if not set (legacy index)."""
    version_file = Path(index_dir) / ".schema_version"
    if version_file.exists():
        return version_file.read_text().strip()
    return "v1_word"


def set_index_version(index_dir: Path, version: str) -> None:
    """Write the index schema version."""
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    (Path(index_dir) / ".schema_version").write_text(version)


def index_file(path: Path, collection: chromadb.Collection, embedder) -> dict:
    """Index a single file. Returns metadata dict."""
    sys.path.insert(0, str(Path(__file__).parent))
    import config as cfg

    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {path.suffix}")

    conf = cfg.load()
    chunk_size = conf.get("chunk_size_tokens", 200)
    chunk_overlap = conf.get("chunk_overlap_tokens", 20)
    batch_size = conf.get("batch_size", 64)

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
        for chunk in chunk_text(page_text, chunk_size=chunk_size, overlap=chunk_overlap):
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
    set_index_version(index_dir, INDEX_VERSION)
    return results


def update_book_topics(filename: str, topics: list[str], collection: chromadb.Collection) -> int:
    """Overwrite the 'topics' field on every chunk for a book. Returns chunk count."""
    existing = collection.get(where={"file": filename}, include=["metadatas"])
    if not existing["ids"]:
        return 0
    topics_json = json.dumps(topics)
    new_metas = [{**m, "topics": topics_json} for m in existing["metadatas"]]
    collection.update(ids=existing["ids"], metadatas=new_metas)
    return len(existing["ids"])


def tag_all_books(books_dir: Path, index_dir: Path) -> list[dict]:
    """Run both LLM and clustering topic detection on all indexed books.

    Returns a comparison list. Stores the best available result per book:
    LLM topics when reachable, clustering topics as fallback.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    import config as cfg
    from topics import detect_topics_clustering, detect_topics_llm, get_book_intro_text

    conf = cfg.load()
    max_chars = conf.get("topics_max_chars", 2500)

    collection = get_collection(index_dir)
    books = list_indexed_books(index_dir)
    if not books:
        return []

    book_files = [b["file"] for b in books]
    book_map = {b["file"]: b for b in books}

    # Strategy: clustering (always runs, offline-safe)
    try:
        cluster_topics = detect_topics_clustering(book_files, collection, books_dir, max_chars=max_chars)
        cluster_error = None
    except Exception as e:
        cluster_topics = {f: [] for f in book_files}
        cluster_error = str(e)

    results = []
    for fname in book_files:
        book = book_map[fname]
        path = books_dir / fname

        # Strategy: LLM (requires connectivity)
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
                "topics": parse_topics(meta.get("topics", "")),
            }
    return list(seen.values())


def search(
    query: str,
    index_dir: Path,
    embedder,
    n: int = 5,
    topic: str | None = None,
) -> tuple[list[dict], dict]:
    """Semantic search. Returns (hits, meta) where meta contains filter diagnostics.

    If topic is given, restricts search to books tagged with that topic.
    If the topic filter matches nothing, falls back to unfiltered search and
    sets meta['filter_ignored'] = True so callers can surface the fallback.
    """
    collection = get_collection(index_dir)
    total = collection.count()
    if total == 0:
        return [], {"filter_ignored": False, "filter_topic": topic}

    where = None
    filter_ignored = False

    if topic:
        all_items = collection.get(include=["metadatas"])
        matching_files: set[str] = set()
        for meta in all_items["metadatas"]:
            stored_topics = parse_topics(meta.get("topics", ""))
            if any(topic.lower() in t.lower() for t in stored_topics):
                matching_files.add(meta["file"])
        if matching_files:
            where = {"file": {"$in": list(matching_files)}}
        else:
            filter_ignored = True
            print(
                f"[library-search] WARNING: topic filter '{topic}' matched no documents. "
                "Falling back to unfiltered search.",
                flush=True,
            )

    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n, total),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
    except Exception:
        filter_ignored = True
        print(
            f"[library-search] WARNING: topic filter '{topic}' query failed. "
            "Falling back to unfiltered search.",
            flush=True,
        )
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
            "topics": parse_topics(meta.get("topics", "")),
            "score": round(1 - dist, 3),
            "passage": doc,
        })

    return hits, {"filter_ignored": filter_ignored, "filter_topic": topic}
