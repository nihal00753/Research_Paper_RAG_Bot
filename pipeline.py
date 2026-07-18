"""
pipeline.py — the seam between retrieval, generation, and anything that
consumes this project (the Streamlit app, or the separate RAGAS eval project
via an adapter that just imports `run`).

Named configs make the four-way ablation (naive / +hybrid / +reranker /
full stack) a one-line change instead of five kwargs repeated everywhere.
"""

import time

import generate
import retrieve

CONFIGS = {
    "naive": {
        "use_hybrid": False,
        "query_rewrite": None,
        "use_reranker": False,
        "use_parent_expansion": False,
        "top_k": 5,
    },
    "hybrid": {
        "use_hybrid": True,
        "query_rewrite": None,
        "use_reranker": False,
        "use_parent_expansion": False,
        "top_k": 5,
    },
    "hybrid_reranked": {
        "use_hybrid": True,
        "query_rewrite": None,
        "use_reranker": True,
        "use_parent_expansion": True,
        "top_k": 5,
    },
    "full_stack_hyde": {
        "use_hybrid": True,
        "query_rewrite": "hyde",
        "use_reranker": True,
        "use_parent_expansion": True,
        "top_k": 5,
    },
    "full_stack_multiquery": {
        "use_hybrid": True,
        "query_rewrite": "multi_query",
        "use_reranker": True,
        "use_parent_expansion": True,
        "top_k": 5,
    },
}


def run(query: str, config: str | dict = "hybrid_reranked") -> dict:
    """
    config: either a name from CONFIGS, or a raw config dict (so the eval
    project can sweep arbitrary combinations without editing this file).
    """
    cfg = CONFIGS[config] if isinstance(config, str) else config

    t0 = time.time()
    retrieval = retrieve.retrieve(query, cfg)
    t_retrieval = time.time() - t0

    t1 = time.time()
    answer = generate.generate_answer(query, retrieval["final_parents"])
    t_generation = time.time() - t1

    return {
        "query": query,
        "config": config,
        "answer": answer,
        "stages": retrieval["stages"],
        "final_parents": retrieval["final_parents"],
        "timing": {
            "retrieval_s": round(t_retrieval, 2),
            "generation_s": round(t_generation, 2),
            "total_s": round(t_retrieval + t_generation, 2),
        },
    }
