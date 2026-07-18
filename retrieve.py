import json
import os
import pickle
from pathlib import Path

import faiss
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

import generate

load_dotenv()

INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

DEFAULT_CONFIG = {
    "use_hybrid": True,
    "query_rewrite": None,       # None | "hyde" | "multi_query"
    "use_reranker": True,
    "use_parent_expansion": True,
    "top_k": 5,
}


class RetrievalIndex:
    """Loads FAISS + BM25 + chunk store once; reused across queries."""

    def __init__(self):
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self.faiss_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            self.bm25_index = pickle.load(f)
        chunk_store = json.loads((INDEX_DIR / "chunks.json").read_text())
        self.parents = chunk_store["parents"]     # dict: parent_id -> parent chunk
        self.children = chunk_store["children"]   # list, index position == FAISS row

    # -- primitive searches -------------------------------------------------

    def dense_search(self, query: str, k: int) -> list[dict]:
        emb = self.embed_model.encode([query], normalize_embeddings=True)
        scores, idxs = self.faiss_index.search(emb, k)
        return [dict(self.children[i], score=float(s))
                for s, i in zip(scores[0], idxs[0]) if i != -1]

    def bm25_search(self, query: str, k: int) -> list[dict]:
        scores = self.bm25_index.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [dict(self.children[i], score=float(scores[i])) for i in top_idx]

    # -- fusion ---------------------------------------------------------------

    def rrf_fuse(self, ranked_lists: list[list[dict]], k_rrf: int = 60,
                 top_k: int = 10) -> list[dict]:
        """Reciprocal Rank Fusion across N ranked chunk lists (dedup by id)."""
        scores: dict[str, float] = {}
        chunk_by_id: dict[str, dict] = {}
        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked):
                cid = chunk["id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
                chunk_by_id[cid] = chunk
        ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [dict(chunk_by_id[cid], rrf_score=scores[cid]) for cid in ordered_ids[:top_k]]

    def hybrid_search(self, query: str, k: int) -> list[dict]:
        dense = self.dense_search(query, k * 2)
        sparse = self.bm25_search(query, k * 2)
        return self.rrf_fuse([dense, sparse], top_k=k)

    # -- reranking + parent expansion ---------------------------------------

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return candidates
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [dict(c, rerank_score=float(s)) for c, s in reranked[:top_k]]

    def expand_to_parents(self, children: list[dict]) -> list[dict]:
        seen, parents = set(), []
        for c in children:
            pid = c["parent_id"]
            if pid not in seen:
                seen.add(pid)
                parents.append(self.parents[pid])
        return parents


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------

_index = None


def get_index() -> RetrievalIndex:
    global _index
    if _index is None:
        _index = RetrievalIndex()
    return _index


def retrieve(query: str, config: dict | None = None) -> dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    idx = get_index()
    stages: dict[str, list[dict]] = {}

    # 1. query rewriting -> one or more search queries
    search_queries = [query]
    if cfg["query_rewrite"] == "hyde":
        hyde_doc = generate.generate_hyde_document(query)
        stages["hyde_document"] = [{"text": hyde_doc}]
        search_queries = [hyde_doc]
    elif cfg["query_rewrite"] == "multi_query":
        variants = generate.generate_query_variants(query)
        stages["query_variants"] = [{"text": v} for v in variants]
        search_queries = [query] + variants

    # 2. retrieval per search query, then fuse across queries too
    per_query_results = []
    for sq in search_queries:
        if cfg["use_hybrid"]:
            per_query_results.append(idx.hybrid_search(sq, cfg["top_k"] * 2))
        else:
            per_query_results.append(idx.dense_search(sq, cfg["top_k"] * 2))

    candidates = (idx.rrf_fuse(per_query_results, top_k=cfg["top_k"] * 2)
                  if len(per_query_results) > 1 else per_query_results[0])
    stages["retrieved"] = candidates

    # 3. rerank
    if cfg["use_reranker"]:
        candidates = idx.rerank(query, candidates, top_k=cfg["top_k"])
        stages["reranked"] = candidates
    else:
        candidates = candidates[: cfg["top_k"]]

    # 4. parent-doc expansion (small chunks found it, large chunks answer from it)
    if cfg["use_parent_expansion"]:
        final_parents = idx.expand_to_parents(candidates)
    else:
        final_parents = candidates

    return {
        "stages": stages,
        "final_children": candidates,
        "final_parents": final_parents,
    }
