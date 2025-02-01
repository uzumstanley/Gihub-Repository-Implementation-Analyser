from typing import Any, List
from uuid import uuid4

import adalflow as adal
from adalflow.core.types import (
    Conversation,
    DialogTurn,
    UserQuery,
    AssistantResponse,
)
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.components.data_process import (
    RetrieverOutputToContextStr,
)
from adalflow.core.component import DataComponent
from config import configs
from src.data_pipeline import DatabaseManager
from adalflow.utils import printc


class Memory(DataComponent):
    """Simple conversation management with a list of dialog turns."""

    def __init__(self):
        super().__init__()
        self.current_conversation = Conversation()

    def call(self) -> List[DialogTurn]:

        all_dialog_turns = self.current_conversation.dialog_turns

        return all_dialog_turns

    def add_dialog_turn(self, user_query: str, assistant_response: str):
        dialog_turn = DialogTurn(
            id=str(uuid4()),
            user_query=UserQuery(query_str=user_query),
            assistant_response=AssistantResponse(response_str=assistant_response),
        )

        self.current_conversation.append_dialog_turn(dialog_turn)


system_prompt = r"""
You are a code assistant which answer's user question on a Github Repo. 
You will receive user query, relevant context, and past conversation history.
Think step by step."""

# history is a list of dialog turns
RAG_TEMPLATE = r"""<START_OF_SYS_PROMPT>
{{system_prompt}}
{{output_format_str}}
<END_OF_SYS_PROMPT>
<START_OF_CONVERSATION_HISTORY>
{% for dialog_turn in conversation_history %}
{{loop.index }}. 
User: {{dialog_turn.user_query.query_str}}
You: {{dialog_turn.assistant_response.response_str}}
{% endfor %}
<END_OF_CONVERSATION_HISTORY>
<START_OF_CONTEXT>
{{context_str}}
<END_OF_CONTEXT>
<START_OF_USER_PROMPT>
{{input_str}}
<END_OF_USER_PROMPT>
"""

from dataclasses import dataclass, field


@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str = field(default="", metadata={"desc": "Rationale for the answer."})
    answer: str = field(default="", metadata={"desc": "Answer to the user query."})

    __output_fields__ = ["rationale", "answer"]


class RAG(adal.Component):
    __doc__ = """RAG with one repo.
    If you want to load a new repo. You need to call prepare_retriever(repo_url_or_path) first."""

    def __init__(self):

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
        data_parser = adal.DataClassParser(data_class=RAGAnswer, return_data_class=True)

        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": data_parser.get_output_format_str(),
                "conversation_history": self.memory(),
                "system_prompt": system_prompt,
                "context_str": None,
            },
            model_client=configs["generator"]["model_client"](),
            model_kwargs=configs["generator"]["model_kwargs"],
            output_processors=data_parser,
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

    def call(self, query: str) -> Any:

        retrieved_documents = self.retriever(query)
        # fill in the document
        for i, retriever_output in enumerate(retrieved_documents):
            retrieved_documents[i].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retriever_output.doc_indices
            ]

        context_str = self.retriever_output_processors(retrieved_documents)
        prompt_kwargs = {
            "input_str": query,
            "context_str": context_str,
            "conversation_history": self.memory(),
        }
        response = self.generator(
            prompt_kwargs=prompt_kwargs,
        )

        prompt_str = self.generator.get_prompt(**prompt_kwargs)
        printc(f"prompt_str: {prompt_str}")

        final_response = response.data

        self.memory.add_dialog_turn(user_query=query, assistant_response=final_response)

        return final_response, retrieved_documents


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
