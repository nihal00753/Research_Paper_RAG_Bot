
import streamlit as st

import pipeline
import retrieve

st.set_page_config(page_title="Research Paper Assistant", layout="centered")

STRATEGY_LABELS = {
    "naive": "Basic (dense search only)",
    "hybrid": "Hybrid (keyword + semantic search)",
    "hybrid_reranked": "Hybrid + reranked (recommended)",
    "full_stack_hyde": "Hybrid + reranked + HyDE",
    "full_stack_multiquery": "Hybrid + reranked + multi-query",
}


@st.cache_data(show_spinner=False)
def cached_run(query: str, config_name: str):
    return pipeline.run(query, config_name)


# Load the FAISS/BM25/embedding/reranker models once, up front, so the first
# question doesn't pay the load cost mid-click.
retrieve.get_index()

st.title("Research Paper Assistant")
st.caption("Ask a question about the RAG papers in the corpus.")

query = st.text_input(
    "Your question",
    placeholder="e.g. How does HyDE differ from multi-query retrieval?",
)

with st.expander("Options"):
    config_name = st.selectbox(
        "Search strategy",
        options=list(STRATEGY_LABELS.keys()),
        format_func=lambda k: STRATEGY_LABELS[k],
        index=list(STRATEGY_LABELS.keys()).index("hybrid_reranked"),
    )
    show_details = st.checkbox("Show retrieval details (for the curious)")

if st.button("Ask", type="primary") and query:
    with st.spinner("Thinking..."):
        result = cached_run(query, config_name)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    for p in result["final_parents"]:
        with st.expander(p["paper_id"]):
            st.text(p["text"][:500] + ("..." if len(p["text"]) > 500 else ""))

    if show_details:
        st.markdown("---")
        st.subheader("Retrieval, stage by stage")
        st.caption(f"Strategy used: {STRATEGY_LABELS[config_name]}")
        for stage_name, chunks in result["stages"].items():
            with st.expander(f"{stage_name} ({len(chunks)})"):
                for c in chunks:
                    label = c.get("paper_id", "")
                    score = c.get("rerank_score")
                    if score is None:
                        score = c.get("rrf_score")
                    if score is None:
                        score = c.get("score")
                    score_str = f" · score {score:.3f}" if score is not None else ""
                    st.markdown(f"**[{label}]**{score_str}")
                    st.text(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
        t = result["timing"]
        st.caption(
            f"Retrieval: {t['retrieval_s']}s · Generation: {t['generation_s']}s · "
            f"Total: {t['total_s']}s"
        )
elif not query:
    st.info("Type a question above and press Ask.")