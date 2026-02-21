#!/usr/bin/env python3
"""Compare substring search vs semantic search for CDAWeb catalog.

Loads the full CDAWeb catalog, builds a fastembed embedding index,
and runs a set of test queries through both search methods to compare
result quality and timing.

Usage:
    venv/bin/python scripts/compare_catalog_search.py
"""

import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so knowledge/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_data_dir
from knowledge.catalog_search import get_full_catalog, search_catalog

EMBEDDINGS_CACHE = get_data_dir() / "cdaweb_embeddings.npy"
TEXTS_CACHE = get_data_dir() / "cdaweb_embed_texts.npy"

# Test queries: mix of exact-match-friendly and natural-language
TEST_QUERIES = [
    "Voyager temperature plasma interstellar",
    "Voyager plasma",
    "solar wind proton density",
    "magnetic field Mars",
    "electron flux radiation belt",
    "ACE magnetometer",
    "thermal ion composition",
    "interplanetary shock",
]

TOP_K = 10  # Number of semantic results to return


def build_embeddings(texts: list[str]) -> np.ndarray:
    """Build or load cached fastembed embeddings for catalog texts."""
    if EMBEDDINGS_CACHE.exists() and TEXTS_CACHE.exists():
        cached_texts = np.load(TEXTS_CACHE, allow_pickle=True)
        if len(cached_texts) == len(texts) and all(
            a == b for a, b in zip(cached_texts, texts)
        ):
            print(f"  Loading cached embeddings from {EMBEDDINGS_CACHE}")
            return np.load(EMBEDDINGS_CACHE)

    from fastembed import TextEmbedding

    print("  Initializing fastembed model (BAAI/bge-small-en-v1.5)...")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print(f"  Embedding {len(texts)} catalog entries...")
    t0 = time.perf_counter()
    embeddings = np.array(list(model.embed(texts, batch_size=256)))
    elapsed = time.perf_counter() - t0
    print(f"  Embedding complete in {elapsed:.1f}s — shape: {embeddings.shape}")

    # Cache
    EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_CACHE, embeddings)
    np.save(TEXTS_CACHE, np.array(texts, dtype=object))
    print(f"  Cached embeddings to {EMBEDDINGS_CACHE}")

    return embeddings


def semantic_search(
    query: str,
    model,
    catalog_embeddings: np.ndarray,
    catalog: list[dict],
    top_k: int = TOP_K,
) -> list[tuple[dict, float]]:
    """Embed query and find top-k most similar catalog entries."""
    query_emb = np.array(list(model.embed([query])))[0]

    # Cosine similarity (embeddings are already normalized by bge)
    similarities = catalog_embeddings @ query_emb

    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        entry = catalog[idx]
        score = similarities[idx]
        results.append(({"id": entry["id"], "title": entry["title"]}, float(score)))
    return results


def print_comparison(
    query: str,
    substring_results: list[dict],
    semantic_results: list[tuple[dict, float]],
    substring_time_ms: float,
    semantic_time_ms: float,
):
    """Print a side-by-side comparison for one query."""
    print(f"\n{'='*100}")
    print(f"  QUERY: \"{query}\"")
    print(f"{'='*100}")

    print(f"\n  SUBSTRING SEARCH ({len(substring_results)} results, {substring_time_ms:.1f}ms)")
    print(f"  {'-'*60}")
    if not substring_results:
        print("  (no results)")
    else:
        for i, r in enumerate(substring_results[:TOP_K], 1):
            print(f"  {i:2d}. {r['id']}")
            print(f"      {r['title'][:80]}")

    print(f"\n  SEMANTIC SEARCH (top {TOP_K}, {semantic_time_ms:.1f}ms)")
    print(f"  {'-'*60}")
    for i, (r, score) in enumerate(semantic_results, 1):
        print(f"  {i:2d}. [{score:.3f}] {r['id']}")
        print(f"      {r['title'][:80]}")


def main():
    print("CDAWeb Catalog Search Comparison: Substring vs Semantic\n")

    # 1. Load catalog
    print("Step 1: Loading CDAWeb catalog...")
    t0 = time.perf_counter()
    catalog = get_full_catalog()
    print(f"  Loaded {len(catalog)} entries in {time.perf_counter() - t0:.1f}s")

    if not catalog:
        print("ERROR: No catalog data available. Run the agent once to populate the cache.")
        sys.exit(1)

    # 2. Build text representations
    texts = [f"{e.get('id', '')} {e.get('title', '')}" for e in catalog]

    # 3. Build/load embeddings
    print("\nStep 2: Building embedding index...")
    embeddings = build_embeddings(texts)

    # 4. Initialize model for query embedding
    print("\nStep 3: Preparing query model...")
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 5. Run comparisons
    print("\nStep 4: Running search comparisons...")

    total_substring_ms = 0.0
    total_semantic_ms = 0.0

    for query in TEST_QUERIES:
        # Substring search
        t0 = time.perf_counter()
        substring_results = search_catalog(query, max_results=TOP_K)
        substring_ms = (time.perf_counter() - t0) * 1000

        # Semantic search
        t0 = time.perf_counter()
        semantic_results = semantic_search(query, model, embeddings, catalog, TOP_K)
        semantic_ms = (time.perf_counter() - t0) * 1000

        total_substring_ms += substring_ms
        total_semantic_ms += semantic_ms

        print_comparison(query, substring_results, semantic_results, substring_ms, semantic_ms)

    # Summary
    print(f"\n{'='*100}")
    print(f"  SUMMARY")
    print(f"{'='*100}")
    print(f"  Catalog size:           {len(catalog)} datasets")
    print(f"  Embedding dimensions:   {embeddings.shape[1]}")
    print(f"  Queries tested:         {len(TEST_QUERIES)}")
    print(f"  Avg substring time:     {total_substring_ms / len(TEST_QUERIES):.1f}ms")
    print(f"  Avg semantic time:      {total_semantic_ms / len(TEST_QUERIES):.1f}ms")

    # Count queries where substring returned 0 but semantic found results
    zero_substring = sum(
        1 for q in TEST_QUERIES if not search_catalog(q, max_results=1)
    )
    print(f"  Queries with 0 substring results: {zero_substring}/{len(TEST_QUERIES)}")


if __name__ == "__main__":
    main()
