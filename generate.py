import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_llm(prompt: str, system: str | None = None, temperature: float = 0.3,
             max_tokens: int = 800) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set — copy .env.example to .env and add your key "
            "(https://openrouter.ai/keys)."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


ANSWER_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about a corpus of arXiv "
    "papers on retrieval-augmented generation. Answer only from the provided "
    "context. If the context doesn't contain the answer, say so plainly instead "
    "of guessing. Cite the paper ID (e.g. [2005.11401]) after any claim you draw "
    "from it."
)


def build_context_block(chunks: list[dict]) -> str:
    """chunks: list of {paper_id, text} — the parent chunks retrieved."""
    parts = []
    for c in chunks:
        parts.append(f"[{c['paper_id']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return call_llm(prompt, system=ANSWER_SYSTEM_PROMPT, temperature=0.2)


def generate_hyde_document(query: str) -> str:
    """HyDE: ask the LLM to write a hypothetical passage that would answer the
    query, as if it were an excerpt from a RAG paper. We embed THIS instead of
    the raw query — it's denser in the vocabulary papers actually use."""
    prompt = (
        f"Write a short (3-4 sentence) passage, in the style of an arXiv paper "
        f"on retrieval-augmented generation, that would answer this question:\n\n"
        f"{query}"
    )
    return call_llm(prompt, temperature=0.5, max_tokens=200)


def generate_query_variants(query: str, n: int = 4) -> list[str]:
    """Multi-query / RAG-Fusion: ask for n reworded variants of the query,
    each surfacing a different angle, to widen retrieval before RRF fusion."""
    prompt = (
        f"Generate {n} different search queries that explore different angles "
        f"of this question. One per line, no numbering, no extra text:\n\n{query}"
    )
    raw = call_llm(prompt, temperature=0.7, max_tokens=200)
    variants = [line.strip("-• ").strip() for line in raw.splitlines() if line.strip()]
    return variants[:n] or [query]
