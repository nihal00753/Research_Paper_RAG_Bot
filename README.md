# RAG Research Paper Q&A Assistant

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Vector Store](https://img.shields.io/badge/vector%20store-FAISS-brightgreen)
![Search](https://img.shields.io/badge/search-BM25%20%2B%20Dense-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![LLM](https://img.shields.io/badge/inference-OpenRouter-9cf)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A retrieval-augmented QA system built over a corpus of arXiv papers on retrieval-augmented generation itself — hybrid search, query rewriting (HyDE, multi-query/RAG-Fusion), cross-encoder reranking, and parent-document compression, wired into a Streamlit app that answers questions against the papers directly.

Most RAG portfolio projects stop at "chatbot over some PDFs." This one is built to demonstrate *why* each retrieval technique exists — every strategy is an independent, swappable config, not a single hardcoded chain.

## What this project does

Ask a question like *"How does HyDE differ from multi-query retrieval?"* and it:
1. Rewrites the query if configured to (HyDE generates a hypothetical answer passage and searches with that; multi-query generates several reworded variants and fuses the results)
2. Runs BM25 (keyword) and dense (semantic) search in parallel, fused with Reciprocal Rank Fusion
3. Reranks the merged candidates with a cross-encoder
4. Expands the surviving small "child" chunks back to their full "parent" chunks for context
5. Answers strictly from that context, citing the source arXiv ID

Every one of those steps is a toggle, not a fixed pipeline — see `pipeline.py`.

## How it was built

| Stage | File | What happens |
|---|---|---|
| Corpus | `ingest.py` | Category-balanced arXiv API queries (not a hand-typed list) build the fetch list; PDFs are downloaded and text-extracted; each paper is split into small "child" chunks (searched) and large "parent" chunks (returned to the LLM), linked by ID |
| Indexing | `ingest.py` | Child chunks embedded with BGE (open-source, local — no API cost) into a FAISS index; a BM25 index is built over the same chunks for keyword search |
| Retrieval | `retrieve.py` | Hybrid search, HyDE, multi-query/RAG-Fusion, cross-encoder reranking, and parent expansion — all behind one `retrieve(query, config)` call that returns every intermediate stage, not just the final result |
| Generation | `generate.py` | Thin OpenRouter wrapper — also used for query rewriting, not just the final answer |
| Orchestration | `pipeline.py` | Five named configs (`naive` → `hybrid` → `hybrid_reranked` → `full_stack_hyde` / `full_stack_multiquery`) so the ablation used for evaluation is a one-line change |
| Interface | `app.py` | Streamlit app — plain by default (question, answer, sources), with an opt-in "show retrieval details" view for anyone who wants to see the mechanics stage by stage |

## Limitations

- **Corpus size**: ~50-100 papers is enough to demonstrate the retrieval techniques, not enough to be a comprehensive RAG literature tool.
- **Free-tier LLM variance**: `openrouter/free` auto-routes to whichever free model is live, which trades reliability for zero cost — answer quality and latency will vary run to run. A pinned model is one env-var change away (see `.env.example`).
- **PDF extraction quality**: `pypdf` text extraction degrades on papers with heavy math notation, multi-column layouts, or figures/tables — those sections often extract as garbled or missing text.
- **CPU-bound inference**: embedding and reranking run locally on CPU by default; ingestion of ~80 papers takes several minutes, and there's no GPU acceleration path built in yet.
- **No incremental ingestion**: `ingest.py` rebuilds the full corpus and both indices from scratch every run — adding one new paper means re-embedding everything.
- **Single-turn QA**: no conversation memory — each question is independent, there's no follow-up/clarification handling.
- **Evaluation lives elsewhere**: RAGAS metrics (faithfulness, context precision/recall, answer relevancy) across the five configs are in a separate companion project, not this repo — see below.

## Future improvements

- Incremental ingestion (only fetch/embed new papers, not a full rebuild)
- Pin and benchmark a fixed LLM instead of the free auto-router, once cost allows
- Add OCR fallback for figure/table-heavy papers
- Multi-turn conversation support (follow-up questions against the same retrieved context)
- Pull the RAGAS comparison table from the companion eval project into this README once it's complete
- Add a basic test suite for the chunking and fusion logic

## Evaluation (companion project)

Retrieval-quality and faithfulness metrics across the configs in `pipeline.py` live in a separate repo, `rag-eval-framework`, which imports `pipeline.run` as an adapter so no logic is duplicated between the two.

## Setup

```bash
git clone <repo-url>
cd rag-paper-assistant
pip install -r requirements.txt
cp .env.example .env   # add your OpenRouter key
```

## Usage

```bash
python ingest.py       # builds data/papers/ and data/index/ (run once; regenerable, gitignored)
streamlit run app.py
```

## Stack

FAISS, BM25 (`rank-bm25`), `sentence-transformers` (BGE embeddings + BGE cross-encoder reranker), OpenRouter, Streamlit.
