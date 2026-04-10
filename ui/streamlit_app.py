import streamlit as st
import requests

st.title("RAG System Demo")
st.caption("Powered by LangChain + Ollama + FAISS")

query = st.text_input("Ask a question:")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        try:
            response = requests.get(
                "http://localhost:8000/ask",
                params={"query": query},
                timeout=60
            )
            data = response.json()
            st.subheader("Answer")
            st.write(data["answer"])
            st.subheader("Evaluation")
            st.json(data["evaluation"])
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach backend. Make sure FastAPI is "
                "running on port 8000."
            )
        except Exception as e:
            st.error(f"Error: {e}")