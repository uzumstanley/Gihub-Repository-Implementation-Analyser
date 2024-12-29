import streamlit as st
import os
import tempfile
from rag import RAG
from adalflow.core.types import Document
from data_pipeline import (
    extract_class_definition,
    extract_class_name_from_query,
    read_all_documents,
    transform_documents_and_save_to_db
)

# Initialize RAG system with repository data
@st.cache_resource
def init_rag(_repo_path: str):
    """Initialize RAG with repository data. Using _repo_path for cache key."""
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
    # Create a temporary directory for the database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "repo_db")
    
    # Process repository files
    with st.spinner("Processing repository files..."):
        documents = read_all_documents(_repo_path)
        if not documents:
            st.error("No documents found in the repository!")
            return None
    
    # Initialize database with repository documents
    with st.spinner("Creating embeddings..."):
        transform_documents_and_save_to_db(documents, db_path)
    
    # Create RAG instance with repository database
    return RAG(index_path=db_path)

st.title("Repository Code Assistant")
st.caption("Analyze and ask questions about your code repository")

# Repository path input
repo_path = st.text_input(
    "Repository Path",
    value=os.getcwd(),
    help="Enter the full path to your repository"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None

# Initialize RAG with repository path
if st.button("Load Repository"):
    if os.path.exists(repo_path):
        st.session_state.rag = init_rag(repo_path)
        if st.session_state.rag:
            st.success(f"Repository loaded successfully from: {repo_path}")
    else:
        st.error("Invalid repository path!")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    if st.session_state.rag:
        st.session_state.rag.memory.current_conversation.dialog_turns.clear()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "context" in message:
            with st.expander(f"View source from {message.get('file_path', 'unknown')}"):
                st.code(message["context"], language=message.get("language", "python"))

# Chat input
if st.session_state.rag and (prompt := st.chat_input(
    "Ask about the code (e.g., 'Show me the implementation of the RAG class', 'How is memory handled?')"
)):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Check if user is asking about a specific class
    class_name = extract_class_name_from_query(prompt)
    
    # Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing code..."):
            response, docs = st.session_state.rag(prompt)
            
            # Show relevant context first, then the explanation
            if docs and docs[0].documents:
                # Try to find implementation code first
                implementation_docs = [
                    doc for doc in docs[0].documents 
                    if doc.meta_data.get("is_implementation", False)
                ]
                
                # Use implementation if found, otherwise use first document
                doc = implementation_docs[0] if implementation_docs else docs[0].documents[0]
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
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "context": context,
                    "file_path": file_path,
                    "language": file_type
                })
            else:
                st.write(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
elif not st.session_state.rag:
    st.info("Please load a repository first!") 