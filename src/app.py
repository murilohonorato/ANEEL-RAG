"""
app.py — Interface Streamlit do ANEEL RAG
Execute com: streamlit run src/app.py
"""

import importlib.util
import sys
from pathlib import Path

import streamlit as st
from qdrant_client import QdrantClient

st.set_page_config(
    page_title="ANEEL RAG",
    page_icon="⚡",
    layout="centered",
)

# ── Carregamento dos recursos ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando modelos, aguarde...")
def load_resources():
    spec = importlib.util.spec_from_file_location(
        "query",
        Path(__file__).parent / "06_query.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["query"] = mod
    spec.loader.exec_module(mod)

    client = QdrantClient(path=str(Path(__file__).parent.parent / "qdrant_db"))
    return mod, client


# ── Interface ─────────────────────────────────────────────────────────────────

st.title("⚡ ANEEL RAG")
st.caption("Consulta à legislação do setor elétrico brasileiro")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            try:
                mod, client = load_resources()
                result = mod.query_pipeline(question, client=client)
                answer = result.get("answer", "Não foi possível gerar uma resposta.")
            except Exception as exc:
                answer = f"❌ Erro: {exc}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
