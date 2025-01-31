import streamlit as st
import os
from src.rag import RAG
from src.data_pipeline import (
    extract_class_definition,
    extract_class_name_from_query,
)

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

if st.button("Clear Chat"):
    st.session_state.messages = []
    if st.session_state.rag:
        st.session_state.rag.memory.current_conversation.dialog_turns.clear()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "context" in message:
            with st.expander(f"View source from {message.get('file_path', 'unknown')}"):
                st.code(message["context"], language=message.get("language", "python"))

if st.session_state.rag and (
    prompt := st.chat_input(
        "Ask about the code (e.g., 'Show me the implementation of the RAG class', 'How is memory handled?')"
    )
):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    class_name = extract_class_name_from_query(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing code..."):
            response, docs = st.session_state.rag(prompt)

            # Show relevant context first, then the explanation
            if docs and docs[0].documents:
                # Try to find implementation code first
                implementation_docs = [
                    doc
                    for doc in docs[0].documents
                    if doc.meta_data.get("is_implementation", False)
                ]

                # Use implementation if found, otherwise use first document
                doc = (
                    implementation_docs[0]
                    if implementation_docs
                    else docs[0].documents[0]
                )
                context = doc.text
                file_path = doc.meta_data.get("file_path", "unknown")
                file_type = doc.meta_data.get("type", "python")

                # If asking about a specific class, try to extract just that class definition
                if class_name and file_type == "python":
                    class_context = extract_class_definition(context, class_name)
                    if class_context != context:  # Only use if we found the class
                        context = class_context

                with st.expander(f"View source from {file_path}"):
                    st.code(context, language=file_type)

                # Now show the explanation
                st.write(response)

                # Add to chat history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "context": context,
                        "file_path": file_path,
                        "language": file_type,
                    }
                )
            else:
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
elif not st.session_state.rag:
    st.info("Please load a repository first!")
