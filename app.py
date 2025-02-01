import streamlit as st
import os
from src.rag import RAG

from typing import List

from config import DEFAULT_GITHUB_REPO


def init_rag(repo_path_or_url: str):

    # from adalflow.utils import setup_env

    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    rag = RAG()
    print(f"Loading repository from: {repo_path_or_url}")
    rag.prepare_retriever(repo_url_or_path=repo_path_or_url)
    return rag


st.title("GithubChat")
st.caption("Learn a repo with RAG assistant")

repo_path = st.text_input(
    "Repository Path",
    value=DEFAULT_GITHUB_REPO,
    help="Github repo URL",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None

if st.button("Initialize local RAG"):
    try:
        st.session_state.rag = init_rag(repo_path)
        if st.session_state.rag:
            st.toast("Repository loaded successfully!")
    except Exception as e:
        st.toast(f"Load failed for repository at: {repo_path}")

# TODO: Better reset the conversation
if st.button("Clear Chat"):
    st.session_state.messages = []
    if st.session_state.rag:
        st.session_state.rag.memory.current_conversation.dialog_turns.clear()


def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write("Assistant:")
            if "rationale" in message:
                st.write(message["rationale"])
            st.write(message["content"])
            if "context" in message:
                with st.expander("View context"):
                    for doc in message["context"]:
                        st.write(
                            f"file_path: {doc.meta_data.get('file_path', 'unknown')}"
                        )
                        st.write(f"language: {doc.meta_data.get('type', 'unknown')}")
                        language = doc.meta_data.get("type", "python")
                        if language == "py":
                            st.code(doc.text, language="python")
                        else:
                            st.write(doc.text)


from adalflow.core.types import Document


def form_context(context: List[Document]):
    formatted_context = ""
    for doc in context:
        formatted_context += ""
        f"file_path: {doc.meta_data.get('file_path', 'unknown')} \n"
        f"language: {doc.meta_data.get('type', 'python')} \n"
        f"content: {doc.text} \n"
    return formatted_context


if st.session_state.rag and (
    query := st.chat_input(
        "Ask about the code (e.g., 'Show me the implementation of the RAG class', 'How is memory handled?')"
    )
):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing code..."):

            st.write(f"memory: {st.session_state.rag.memory()}")
            response, docs = st.session_state.rag(query)

            # Show relevant context first, then the explanation
            if docs and docs[0].documents:
                context = docs[0].documents

                # Add to chat history
                st.write(f"add to history")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "rationale": (
                            response.rationale
                            if hasattr(response, "rationale")
                            else None
                        ),
                        "content": (
                            response.answer
                            if hasattr(response, "answer")
                            else response.raw_response
                        ),
                        "context": context,
                    }
                )
            else:
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
elif not st.session_state.rag:
    st.info("Please load a repository first!")

# Finally, call display_messages *after* everything is appended
display_messages()
