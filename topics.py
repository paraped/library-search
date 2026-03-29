"""Topic detection for library books — LLM inference and embedding clustering."""

from __future__ import annotations

import json
import re
from pathlib import Path


def get_book_intro_text(path: Path, pages: int = 3) -> str:
    """Extract plain text from the first N pages of a PDF."""
    import fitz
    doc = fitz.open(str(path))
    parts = []
    for i in range(min(pages, len(doc))):
        text = doc[i].get_text("text").strip()
        if text:
            parts.append(text)
    doc.close()
    return "\n".join(parts)


def detect_topics_llm(title: str, author: str, intro_text: str) -> list[str]:
    """Assign 1-3 topic tags using Anthropic API (if key set) or Claude CLI fallback.

    Priority:
      1. Anthropic SDK with llm_api_key from config
      2. claude CLI (Claude Code users without a separate API key)
    Raises on error — caller falls back to clustering.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import config as cfg

    prompt = (
        f"Title: {title}\n"
        f"Author: {author}\n\n"
        f"Book excerpt (first pages):\n{intro_text[:2500]}\n\n"
        "Assign 1-3 short topic tags to this book. "
        "Examples: 'building', 'conservation', 'history', 'gardening', 'cooking', 'engineering'. "
        "Return ONLY a JSON array of lowercase strings, nothing else."
    )

    conf = cfg.load()
    api_key = conf.get("llm_api_key", "").strip()
    model = conf.get("llm_model", "claude-haiku-4-5-20251001")

    if api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
    else:
        import shutil
        import subprocess
        claude_bin = shutil.which("claude")
        if not claude_bin:
            raise RuntimeError(
                "LLM unavailable: set llm_api_key in Settings, or install Claude Code"
            )
        result = subprocess.run(
            [claude_bin, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"claude exited {result.returncode}")
        text = result.stdout.strip()

    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        raw = json.loads(match.group())
        return [str(t).strip().lower() for t in raw if str(t).strip()]
    return []


def detect_topics_clustering(
    book_files: list[str],
    collection,
    books_dir: Path,
) -> dict[str, list[str]]:
    """Cluster books by their mean chunk embeddings; label clusters via TF-IDF.

    Works entirely offline — no API calls.
    Returns {filename: [topic, ...]} for every book in book_files.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not book_files:
        return {}

    # Gather intro texts for TF-IDF
    intro_texts: dict[str, str] = {}
    for fname in book_files:
        path = books_dir / fname
        intro_texts[fname] = get_book_intro_text(path) if path.exists() else fname

    # Single book — just pull top TF-IDF terms from the book itself
    if len(book_files) == 1:
        fname = book_files[0]
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=200)
            tfidf = vec.fit_transform([intro_texts[fname]])
            names = vec.get_feature_names_out()
            top = tfidf[0].toarray()[0].argsort()[-3:][::-1]
            return {fname: [names[i] for i in top if tfidf[0, i] > 0]}
        except Exception:
            return {fname: []}

    # Multiple books — cluster by mean embedding
    book_embeddings: dict[str, list[float]] = {}
    for fname in book_files:
        result = collection.get(where={"file": fname}, include=["embeddings"])
        embs = result.get("embeddings")
        if embs is not None and len(embs) > 0:
            book_embeddings[fname] = list(np.array(embs).mean(axis=0))

    files_with_emb = [f for f in book_files if f in book_embeddings]

    if len(files_with_emb) < 2:
        # Not enough embeddings — fall back to per-book TF-IDF
        result = {}
        for fname in book_files:
            try:
                vec = TfidfVectorizer(stop_words="english", max_features=200)
                tfidf = vec.fit_transform([intro_texts[fname]])
                names = vec.get_feature_names_out()
                top = tfidf[0].toarray()[0].argsort()[-3:][::-1]
                result[fname] = [names[i] for i in top if tfidf[0, i] > 0]
            except Exception:
                result[fname] = []
        return result

    X = np.array([book_embeddings[f] for f in files_with_emb])

    # Choose number of clusters: roughly 1 per 3 books, min 2 (or 1 if only 2 books that are very similar)
    n = len(files_with_emb)
    n_clusters = max(1, min(n - 1, n // 3 + 1))

    if n_clusters == 1:
        labels = [0] * n
    else:
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        )
        labels = clustering.fit_predict(X).tolist()

    # Build per-cluster combined text for TF-IDF labeling
    cluster_text: dict[int, str] = {}
    for fname, label in zip(files_with_emb, labels):
        cluster_text[label] = cluster_text.get(label, "") + " " + intro_texts.get(fname, "")

    sorted_labels = sorted(cluster_text.keys())
    corpus = [cluster_text[lbl] for lbl in sorted_labels]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", max_features=400, ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-z][a-z0-9\-]{2,}\b",  # lowercase only, min 3 chars
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        cluster_topics: dict[int, list[str]] = {}
        for i, lbl in enumerate(sorted_labels):
            row = tfidf_matrix[i].toarray()[0]
            top_idx = row.argsort()[-5:][::-1]
            # prefer multi-word phrases; skip generic single chars
            topics = [feature_names[j] for j in top_idx if row[j] > 0][:3]
            cluster_topics[lbl] = topics
    except Exception:
        cluster_topics = {lbl: [] for lbl in sorted_labels}

    result = {fname: cluster_topics.get(lbl, []) for fname, lbl in zip(files_with_emb, labels)}
    # Include any books that had no embeddings
    for fname in book_files:
        if fname not in result:
            result[fname] = []
    return result
