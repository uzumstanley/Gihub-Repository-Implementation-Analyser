import adalflow as adal

from adalflow.core.types import ModelClientType

from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os
import subprocess
from config import configs


# TODO: fix the delay in the data pipeline, chunk_size and chunk_overlap


def get_embedder():

    embedder = adal.Embedder(
        model_client=ModelClientType.OPENAI(),
        model_kwargs=configs["embedder"]["model_kwargs"],
    )
    return embedder


def get_data_transformer():

    # batch_size = 100

    # splitter_config = {"split_by": "word", "chunk_size": 500, "chunk_overlap": 100}

    splitter = TextSplitter(**configs["text_splitter"])
    embedder = adal.Embedder(
        model_client=ModelClientType.OPENAI(),
        model_kwargs=configs["embedder"]["model_kwargs"],
    )
    embedder_transformer = ToEmbeddings(
        embedder, batch_size=configs["embedder"]["batch_size"]
    )
    data_transformer = adal.Sequential(splitter, embedder_transformer)
    return data_transformer


def download_github_repo(repo_url, local_path):
    """
    Downloads a GitHub repository to a specified local path.

    Args:
        repo_url (str): The URL of the GitHub repository to clone.
        local_path (str): The local directory where the repository will be cloned.

    Returns:
        str: The output message from the `git` command.
    """
    try:
        # Check if Git is installed
        print(f"local_path: {local_path}")
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Ensure the local path exists
        os.makedirs(local_path, exist_ok=True)

        # Clone the repository
        result = subprocess.run(
            ["git", "clone", repo_url, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return result.stdout.decode("utf-8")

    except subprocess.CalledProcessError as e:
        return f"Error during cloning: {e.stderr.decode('utf-8')}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


def read_all_documents(path):
    """
    Recursively reads all documents in a directory and its subdirectories.

    Args:
        path (str): The root directory path.

    Returns:
        list: A list of strings, where each string is the content of a file.
    """
    documents = []
    pathes = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
                    pathes.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return [
        adal.Document(text=doc, meta_data={"title": path})
        for doc, path in zip(documents, pathes)
    ]


from typing import List


def transform_documents_and_save_to_db(documents: List[adal.Document], db_path: str):
    """
    Transforms a list of documents and saves them to a local database.

    Args:
        documents (list): A list of `Document` objects.
        db_path (str): The path to the local database file.
    """
    # Get the data transformer
    data_transformer = get_data_transformer()
    from adalflow.core.db import LocalDB

    # Save the documents to a local database
    db = LocalDB("microsoft_lomps")
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")
    db.save_state(filepath=db_path)


def load_db(db_path: str):

    db = LocalDB(name="microsoft_lomps")
    db.load_state(filepath=db_path)
    return db


from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.core.db import LocalDB


if __name__ == "__main__":
    from adalflow.utils import get_logger

    adal.setup_env()

    # get_logger()
    repo_url = "https://github.com/microsoft/LMOps"
    from adalflow.utils import get_adalflow_default_root_path

    local_path = os.path.join(get_adalflow_default_root_path(), "LMOps")

    # download_github_repo(repo_url, local_path)

    target_path = os.path.join(local_path, "prompt_optimization")

    documents = read_all_documents(target_path)
    print(len(documents))
    print(documents[0])
    # transformed_documents = get_data_transformer()(documents[0:2])
    # print(len(transformed_documents))
    # print(transformed_documents[0])

    # save to local db
    # from adalflow.core.db import LocalDB

    db = LocalDB("microsft_lomps")
    key = "split_and_embed"
    print(get_data_transformer())
    db.register_transformer(transformer=get_data_transformer(), key=key)
    db.load(documents)
    db.transform(key=key)
    transformed_docs = db.transformed_items[key]
    print(len(transformed_docs))
    print(transformed_docs[0])
    db_path = os.path.join(get_adalflow_default_root_path(), "db_microsft_lomps")
    db.save_state(filepath=db_path)
    # db = load_db(db_path)
