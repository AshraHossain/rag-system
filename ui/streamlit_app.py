import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG System", layout="wide")
st.title("Production RAG System")
st.caption(
    "Hybrid BM25 + Dense Retrieval · "
    "Cross-Encoder Reranking · "
    "RAGas Evaluation"
)

# ── Sidebar: status + document upload ────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API online")
        st.metric("Documents indexed", health.get("docs_loaded", "?"))
    except Exception:
        st.error("API offline — start the FastAPI server first")

    st.divider()
    st.header("Upload Document")
    uploaded = st.file_uploader(
        "Add a .txt or .pdf file to the index",
        type=["txt", "pdf"],
    )
    if uploaded and st.button("Ingest"):
        with st.spinner("Ingesting…"):
            try:
                res = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type,
                        )
                    },
                )
                if res.ok:
                    d = res.json()
                    st.success(
                        f"Added {d['chunks_added']} chunks "
                        f"(total: {d['total_docs']})"
                    )
                else:
                    st.error(res.text)
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API.")

# ── Main query ───────────────────────────────────────────────────────────────
query = st.text_input("Ask a question about your documents:")
if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Retrieving and generating…"):
        try:
            res = requests.get(
                f"{API_URL}/ask",
                params={"query": query},
                timeout=60,
            )
            if not res.ok:
                st.error(f"API error {res.status_code}: {res.text}")
                st.stop()
            data = res.json()

            st.subheader("Answer")
            st.write(data["answer"])

            with st.expander("Source passages"):
                for i, src in enumerate(data.get("sources", []), 1):
                    st.markdown(f"**[{i}]** {src}…")

            with st.expander("Evaluation metrics"):
                ev = data.get("evaluation", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Answer relevance",
                    ev.get("answer_relevance", "—"),
                )
                c2.metric(
                    "Hallucination score",
                    ev.get("hallucination_score", "—"),
                    help="Lower is better (0 = fully grounded)",
                )
                c3.metric(
                    "Context recall",
                    ev.get("context_recall", "—"),
                )
                c4.metric("Docs used", ev.get("num_docs_used", "—"))

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach backend. "
                "Start the FastAPI server with: "
                "`uvicorn app.main:app --reload`"
            )
        except Exception as exc:
            st.error(f"Error: {exc}")
