from typing import Optional, Any, List, Dict
import os
from uuid import uuid4

import adalflow as adal
from adalflow.core.db import LocalDB
from adalflow.core.types import (
    ModelClientType,
    Document,
    Conversation,
    DialogTurn,
    UserQuery,
    AssistantResponse,
)
from adalflow.core.string_parser import JsonParser
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.components.data_process import (
    RetrieverOutputToContextStr,
    ToEmbeddings,
    TextSplitter,
)
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.component import Component
from config import configs, prompts
from src.data_pipeline import DatabaseManager


class Memory(Component):
    def __init__(self, turn_db: LocalDB = None):
        """Initialize the Memory component."""
        super().__init__()
        self.current_conversation = Conversation()
        self.turn_db = turn_db or LocalDB()  # all turns
        self.conver_db = LocalDB()  # a list of conversations

    def call(self) -> str:
        """Returns the current conversation history as a formatted string."""
        if not self.current_conversation.dialog_turns:
            return ""

        formatted_history = []
        for turn in self.current_conversation.dialog_turns.values():
            formatted_history.extend(
                [
                    f"User: {turn.user_query.query_str}",
                    f"Assistant: {turn.assistant_response.response_str}",
                ]
            )
        return "\n".join(formatted_history)

    def add_dialog_turn(self, user_query: str, assistant_response: str):
        """Add a new dialog turn to the current conversation."""
        dialog_turn = DialogTurn(
            id=str(uuid4()),
            user_query=UserQuery(query_str=user_query),
            assistant_response=AssistantResponse(response_str=assistant_response),
        )

        self.current_conversation.append_dialog_turn(dialog_turn)
        self.turn_db.add(
            {"user_query": user_query, "assistant_response": assistant_response}
        )


class RAG(adal.Component):
    __doc__ = """RAG with one repo.
    If you want to load a new repo. You need to call prepare_retriever(repo_url_or_path) first."""

    def __init__(self, prompt_type: str = "code_analysis"):
        """Initialize RAG component.

        Args:
            index_path (str, optional): Path to the index database. Defaults to None.
            prompt_type (str, optional): Type of prompt to use ('code_analysis' or 'general_qa').
                                       Defaults to 'code_analysis'.
        """
        super().__init__()

        # Initialize embedder, generator, and db_manager
        self.memory = Memory()

        self.embedder = adal.Embedder(
            model_client=configs["embedder"]["model_client"](),
            model_kwargs=configs["embedder"]["model_kwargs"],
        )

        self.initialize_db_manager()

        self.retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)

        # Get the appropriate prompt template
        prompt_template = prompts.get(prompt_type, prompts["code_analysis"])

        self.generator = adal.Generator(
            prompt_kwargs={
                "task_desc_str": prompt_template,
            },
            model_client=configs["generator"]["model_client"](),
            model_kwargs=configs["generator"]["model_kwargs"],
            output_processors=JsonParser(),
        )

    def initialize_db_manager(self):
        self.db_manager = DatabaseManager()
        self.transformed_docs = []

    def prepare_retriever(self, repo_url_or_path: str):
        r"""Run prepare_retriever once for each repo."""
        self.initialize_db_manager()
        self.transformed_docs = self.db_manager.prepare_database(repo_url_or_path)
        print(f"len(self.transformed_docs): {len(self.transformed_docs)}")
        self.retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=self.embedder,
            documents=self.transformed_docs,
            document_map_func=lambda doc: doc.vector,
        )

    def generate(self, query: str, context: Optional[str] = None) -> Any:
        if not self.generator:
            raise ValueError("Generator is not set")

        # Modify query to focus on implementation if asking about a class
        if "class" in query.lower() and "implementation" not in query.lower():
            query = f"Show and explain the implementation of the {query}"

        # Add conversation history to context
        full_context = ""
        if context:
            full_context += f"Code to analyze:\n```python\n{context}\n```\n"

        # Get conversation history from memory component
        conversation_history = self.memory()
        if conversation_history:
            full_context += f"\nPrevious conversation:\n{conversation_history}"

        prompt_kwargs = {
            "context_str": full_context,
            "input_str": query,
        }
        response = self.generator(prompt_kwargs=prompt_kwargs)
        return response.data["answer"]

    def call(self, query: str) -> Any:
        # Modify query to focus on implementation if asking about a class
        if "class" in query.lower() and "implementation" not in query.lower():
            search_query = f"class implementation {query}"
        else:
            search_query = query

        retrieved_documents = self.retriever(search_query)
        # fill in the document
        for i, retriever_output in enumerate(retrieved_documents):
            retrieved_documents[i].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retriever_output.doc_indices
            ]

        context_str = self.retriever_output_processors(retrieved_documents)
        response = self.generate(query, context=context_str)

        # Update conversation history in memory
        self.memory.add_dialog_turn(user_query=query, assistant_response=response)

        return response, retrieved_documents


if __name__ == "__main__":
    from adalflow.utils import get_logger

    adal.setup_env()
    # repo_url = "https://github.com/SylphAI-Inc/AdalFlow"
    repo_url = "https://github.com/SylphAI-Inc/GithubChat"
    rag = RAG()
    rag.prepare_retriever(repo_url)
    print(
        f"RAG component initialized for repo: {repo_url}. Type your query below or type 'exit' to quit."
    )

    while True:
        # Get user input

        query = input("Enter your query (or type 'exit' to stop): ")

        # Exit condition
        if query.lower() in ["exit", "quit", "stop"]:
            print("Exiting RAG component. Goodbye!")
            break

        # Process the query
        try:
            response, retrieved_documents = rag(query)
            print(f"\nResponse:\n{response}\n")
            print(f"Retrieved Documents:\n{retrieved_documents}\n")
        except Exception as e:
            print(f"An error occurred while processing the query: {e}")
