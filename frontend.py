import os
import tempfile
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from backend import (
    ACTIVE_PROVIDER,
    Workflow,
    clear_pdf,
    get_pdf_info,
    load_and_index_pdf,
)

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

# ─── Session State Init ──────────────────────────────────────────
# session_id doubles as the LangGraph thread_id: the backend keeps the conversation
# history and this session's PDF index under it, so only the display history lives here.
defaults = {
    "session_id": str(uuid.uuid4()),
    "message_history": [],
    "uploader_key": 0,
    "handled_pdf": None,  # signature of the last uploaded file we processed
    "pdf_error": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

session_id = st.session_state["session_id"]


def reset_chat():
    """Start a fresh conversation: new thread, empty history, no PDF."""
    clear_pdf(session_id)
    st.session_state["session_id"] = str(uuid.uuid4())
    st.session_state["message_history"] = []
    st.session_state["handled_pdf"] = None
    st.session_state["pdf_error"] = None
    st.session_state["uploader_key"] += 1  # resets the file uploader widget


# ─── Sidebar ──────────────────────────────────────────────────────
st.sidebar.title("🤖 RAG Chatbot")
st.sidebar.caption(f"**Model:** {ACTIVE_PROVIDER}")

st.sidebar.header("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file", type="pdf", key=f"pdf_{st.session_state['uploader_key']}"
)

if uploaded_file is not None:
    signature = f"{uploaded_file.name}:{uploaded_file.size}"
    # Re-index whenever a *different* file is uploaded (the old code ignored later uploads).
    if st.session_state["handled_pdf"] != signature:
        st.session_state["handled_pdf"] = signature
        st.session_state["pdf_error"] = None
        tmp_path = None
        with st.sidebar.status("Indexing PDF...", expanded=True) as status:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                pages, chunks = load_and_index_pdf(tmp_path, session_id, uploaded_file.name)
                status.update(label=f"✅ Indexed {pages} pages, {chunks} chunks", state="complete")
            except Exception as exc:
                st.session_state["pdf_error"] = str(exc)
                clear_pdf(session_id)
                status.update(label="❌ Could not index PDF", state="error")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
elif st.session_state["handled_pdf"] is not None:
    # The user removed the file with the "x" button.
    clear_pdf(session_id)
    st.session_state["handled_pdf"] = None
    st.session_state["pdf_error"] = None

pdf_info = get_pdf_info(session_id)
if pdf_info:
    st.sidebar.success(f"📄 {pdf_info['name']} — {pdf_info['pages']} pages, {pdf_info['chunks']} chunks")
elif st.session_state["pdf_error"]:
    st.sidebar.error(f"PDF error: {st.session_state['pdf_error']}")

st.sidebar.divider()
st.sidebar.header("🛠️ Available Tools")
st.sidebar.markdown(
    """
- 🔍 **Web Search** — DuckDuckGo
- 🌐 **Webpage Reader** — open any public URL
- 📄 **RAG** — PDF Q&A with page citations
- 🧮 **Calculator** — safe math expressions
- 🕐 **Time** — any timezone
- 📅 **Date Calculator** — days between / add days
- 🌦️ **Weather** — current + 3-day forecast
- 📏 **Unit Converter** — length, mass, temp, data…
- 💱 **Currency Converter** — live rates (PKR supported)
- 📝 **Text Analyzer** — stats, reading time, keywords
"""
)

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# ─── Main Area ────────────────────────────────────────────────────
st.title("💬 RAG-Powered Chatbot")
st.caption(
    "Ask me anything — I can search the web, read pages, check the weather, convert units and "
    "currencies, do math and dates, and answer questions about your uploaded PDF."
)

for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def ai_stream(user_text: str):
    """Stream only the assistant's final answer tokens (not tool calls or tool output)."""
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 20}
    for chunk, meta in Workflow.stream(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
        stream_mode="messages",
    ):
        if meta.get("langgraph_node") != "chat_node" or not isinstance(chunk, AIMessage):
            continue
        if getattr(chunk, "tool_call_chunks", None) or chunk.tool_calls:
            continue
        if isinstance(chunk.content, str) and chunk.content:
            yield chunk.content


# ─── Chat Input ───────────────────────────────────────────────────
if user_input := st.chat_input("Type your message..."):
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            ai_message = st.write_stream(ai_stream(user_input))
        except Exception as exc:
            ai_message = None
            st.error(
                f"Something went wrong while generating the answer: {exc}\n\n"
                "Check your API key / rate limits and try again."
            )

    if ai_message:
        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
