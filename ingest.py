
import os
import json
import pickle
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import faiss
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data/papers"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Category-balanced arXiv fetch list
# ---------------------------------------------------------------------------

# A handful of foundational papers worth always including, verified by hand
# (arXiv IDs are easy to get subtly wrong when typed from memory — these were
# cross-checked against arxiv.org before being hardcoded here).
ANCHOR_PAPERS = [
    "2005.11401",  # Lewis et al. — original RAG
    "2312.10997",  # Gao et al. — RAG for LLMs: A Survey
    "2212.10496",  # HyDE
    "2402.03367",  # RAG-Fusion
    "2309.15217",  # RAGAS
    "2311.09476",  # ARES
    "2402.03216",  # BGE M3-Embedding
]

# Category -> search queries used against the arXiv API. Multiple queries per
# category so one narrow phrasing doesn't dominate the results.
CATEGORY_QUERIES = {
    "chunking": [
        "semantic chunking retrieval augmented generation",
        "recursive text splitting document retrieval",
    ],
    "hybrid_search": [
        "hybrid search BM25 dense retrieval",
        "sparse dense retrieval fusion",
    ],
    "query_rewriting": [
        "multi-query retrieval large language model",
        "query expansion retrieval augmented generation",
    ],
    "reranking": [
        "cross-encoder reranking retrieval",
        "listwise reranker large language model",
    ],
    "advanced_architectures": [
        "self-reflective retrieval augmented generation",
        "corrective retrieval augmented generation",
        "graph retrieval augmented generation knowledge graph",
        "agentic retrieval augmented generation",
    ],
    "evaluation": [
        "retrieval augmented generation evaluation benchmark",
        "faithfulness hallucination retrieval augmented generation",
    ],
    "long_context": [
        "long context language model retrieval",
        "lost in the middle context window",
    ],
}

PER_QUERY_CAP = int(os.getenv("PER_QUERY_CAP", "6"))
TOTAL_CORPUS_CAP = int(os.getenv("TOTAL_CORPUS_CAP", "80"))

ARXIV_API = "http://export.arxiv.org/api/query"


def search_arxiv(query: str, max_results: int = 6) -> list[dict]:
    """Query the arXiv API and return [{id, title, pdf_url}, ...]."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    resp = requests.get(ARXIV_API, params=params, timeout=30)
    resp.raise_for_status()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)

    out = []
    for entry in root.findall("atom:entry", ns):
        raw_id = entry.find("atom:id", ns).text.strip()
        arxiv_id = raw_id.split("/abs/")[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)  # drop version suffix
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        out.append({"id": arxiv_id, "title": title, "pdf_url": pdf_url})
    return out


def build_fetch_list() -> list[dict]:
    """Category-balanced, deduped list of papers to download."""
    seen: dict[str, dict] = {}

    # anchors first — guaranteed inclusion
    for arxiv_id in ANCHOR_PAPERS:
        seen[arxiv_id] = {"id": arxiv_id, "category": "anchor", "title": None,
                           "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}"}

    for category, queries in CATEGORY_QUERIES.items():
        for q in queries:
            try:
                results = search_arxiv(q, max_results=PER_QUERY_CAP)
            except Exception as e:
                print(f"  [warn] query failed: {q!r} ({e})")
                continue
            for r in results:
                if r["id"] not in seen:
                    r["category"] = category
                    seen[r["id"]] = r
            time.sleep(3)  # be polite to the arXiv API rate limit
            if len(seen) >= TOTAL_CORPUS_CAP:
                break
        if len(seen) >= TOTAL_CORPUS_CAP:
            break

    return list(seen.values())[:TOTAL_CORPUS_CAP]


# ---------------------------------------------------------------------------
# 2. Download + text extraction
# ---------------------------------------------------------------------------

def download_pdf(pdf_url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        resp = requests.get(pdf_url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"  [warn] download failed: {pdf_url} ({e})")
        return False


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------------------------------------------------------------------------
# 3. Parent/child chunking
# ---------------------------------------------------------------------------

def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive-ish splitter: break on paragraph, then sentence, then char."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                # paragraph itself too long: hard-split with overlap
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def chunk_paper(paper_id: str, text: str) -> tuple[list[dict], list[dict]]:
    """Returns (parent_chunks, child_chunks). Each child has a parent_id."""
    parent_texts = split_text(text, chunk_size=2000, overlap=200)
    parents = [
        {"id": f"{paper_id}::p{i}", "paper_id": paper_id, "text": t}
        for i, t in enumerate(parent_texts)
    ]

    children = []
    for parent in parents:
        child_texts = split_text(parent["text"], chunk_size=400, overlap=50)
        for j, ct in enumerate(child_texts):
            children.append({
                "id": f"{parent['id']}::c{j}",
                "parent_id": parent["id"],
                "paper_id": paper_id,
                "text": ct,
            })
    return parents, children


# ---------------------------------------------------------------------------
# 4/5. Indices
# ---------------------------------------------------------------------------

def build_faiss_index(children: list[dict], model_name: str):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in children]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                               normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim via normalized inner product
    index.add(embeddings)
    return index


def build_bm25_index(children: list[dict]):
    tokenized = [c["text"].lower().split() for c in children]
    return BM25Okapi(tokenized)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Building category-balanced fetch list from the arXiv API...")
    fetch_list = build_fetch_list()
    print(f"  -> {len(fetch_list)} papers selected")

    all_parents, all_children, metadata = [], [], []

    for paper in tqdm(fetch_list, desc="Downloading + chunking"):
        pdf_path = DATA_DIR / f"{paper['id'].replace('/', '_')}.pdf"
        if not download_pdf(paper["pdf_url"], pdf_path):
            continue
        try:
            text = extract_text(pdf_path)
        except Exception as e:
            print(f"  [warn] extract failed for {paper['id']}: {e}")
            continue
        if len(text.strip()) < 500:
            continue

        parents, children = chunk_paper(paper["id"], text)
        all_parents.extend(parents)
        all_children.extend(children)
        metadata.append({
            "id": paper["id"],
            "title": paper.get("title"),
            "category": paper.get("category"),
            "n_parent_chunks": len(parents),
            "n_child_chunks": len(children),
        })

    print(f"Corpus: {len(metadata)} papers, {len(all_parents)} parent chunks, "
          f"{len(all_children)} child chunks")

    (DATA_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    chunk_store = {
        "parents": {p["id"]: p for p in all_parents},
        "children": all_children,  # keep as list — index position == FAISS row
    }
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunk_store, indent=2))

    print("Building FAISS index...")
    faiss_index = build_faiss_index(all_children, EMBEDDING_MODEL)
    faiss.write_index(faiss_index, str(INDEX_DIR / "faiss.index"))

    print("Building BM25 index...")
    bm25_index = build_bm25_index(all_children)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    print(f"Done. Indices written to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
