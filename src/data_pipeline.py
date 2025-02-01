import adalflow as adal
from adalflow.core.types import Document, List
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os
import subprocess
import re
import glob
from adalflow.utils import get_adalflow_default_root_path
from adalflow.utils import printc
from config import configs
from adalflow.core.db import LocalDB


def extract_class_definition(content: str, class_name: str) -> str:
    """Extract a complete class definition from the content."""
    lines = content.split("\n")
    class_start = -1
    indent_level = 0

    # Find the class definition start
    for i, line in enumerate(lines):
        if f"class {class_name}" in line:
            class_start = i
            # Get the indentation level of the class
            indent_level = len(line) - len(line.lstrip())
            break

    if class_start == -1:
        return content

    # Collect the entire class definition
    class_lines = [lines[class_start]]
    current_line = class_start + 1

    while current_line < len(lines):
        line = lines[current_line]
        # If we hit a line with same or less indentation, we're out of the class
        if line.strip() and len(line) - len(line.lstrip()) <= indent_level:
            break
        class_lines.append(line)
        current_line += 1

    return "\n".join(class_lines)


def extract_class_name_from_query(query: str) -> str:
    """Extract class name from a query about a class."""
    # Common patterns for asking about classes
    patterns = [
        r"class (\w+)",
        r"the (\w+) class",
        r"what does (\w+) do",
        r"how does (\w+) work",
        r"show me (\w+)",
        r"explain (\w+)",
    ]

    query = query.lower()
    words = query.split()

    # First try to find class name using patterns
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            # Return the first match, capitalized
            return matches[0].capitalize()

    # If no pattern match, look for words that might be class names (capitalized words)
    for word in words:
        # Skip common words
        if word in [
            "the",
            "class",
            "show",
            "me",
            "how",
            "does",
            "what",
            "is",
            "are",
            "explain",
        ]:
            continue
        # Return any word that starts with a capital letter in the original query
        original_words = query.split()
        for original_word in original_words:
            if original_word.lower() == word and original_word[0].isupper():
                return original_word

    return None


def download_github_repo(repo_url: str, local_path: str):
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


def read_all_documents(path: str):
    """
    Recursively reads all documents in a directory and its subdirectories.

    Args:
        path (str): The root directory path.

    Returns:
        list: A list of Document objects with metadata.
    """
    documents = []
    # File extensions to look for, prioritizing code files
    code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

    # Process code files first
    for ext in code_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            if ".venv" in file_path or "node_modules" in file_path:
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, path)

                    # Determine if this is an implementation file
                    is_implementation = (
                        not relative_path.startswith("test_")
                        and not relative_path.startswith("app_")
                        and "test" not in relative_path.lower()
                    )

                    doc = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": True,
                            "is_implementation": is_implementation,
                            "title": relative_path,
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Then process documentation files
    for ext in doc_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            if ".venv" in file_path or "node_modules" in file_path:
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, path)
                    doc = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": False,
                            "is_implementation": False,
                            "title": relative_path,
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return documents


def prepare_data_pipeline():
    """Creates and returns the data transformation pipeline."""
    splitter = TextSplitter(**configs["text_splitter"])
    embedder = adal.Embedder(
        model_client=configs["embedder"]["model_client"](),
        model_kwargs=configs["embedder"]["model_kwargs"],
    )
    embedder_transformer = ToEmbeddings(
        embedder=embedder, batch_size=configs["embedder"]["batch_size"]
    )
    data_transformer = adal.Sequential(
        splitter, embedder_transformer
    )  # sequential will chain together splitter and embedder
    return data_transformer


def transform_documents_and_save_to_db(
    documents: List[Document], db_path: str
) -> LocalDB:
    """
    Transforms a list of documents and saves them to a local database.

    Args:
        documents (list): A list of `Document` objects.
        db_path (str): The path to the local database file.
    """
    # Get the data transformer
    data_transformer = prepare_data_pipeline()

    # Save the documents to a local database
    db = LocalDB()
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    return db


def chat_with_adalflow_lib():
    """
    (1) Download repo: https://github.com/SylphAI-Inc/AdalFlow
    (2) Read all documents in the repo
    (3) Transform the documents using the data pipeline
    (4) Save the transformed documents to a local database
    """
    # Download the repository
    repo_url = "https://github.com/SylphAI-Inc/AdalFlow"
    local_path = os.path.join(get_adalflow_default_root_path(), "AdalFlow")
    download_github_repo(repo_url, local_path)
    # Read all documents in the repository
    documents = read_all_documents(local_path)
    # Transform the documents using the data pipeline
    db_path = os.path.join(get_adalflow_default_root_path(), "db_adalflow")
    transform_documents_and_save_to_db(documents, db_path)


class DatabaseManager:
    """
    Manages the creation, loading, transformation, and persistence of LocalDB instances.
    """

    def __init__(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def prepare_database(self, repo_url_or_path: str) -> List[Document]:
        """
        Create a new database from the repository.
        :return: List of Document objects
        """
        self.reset_database()
        self._create_repo(repo_url_or_path)
        return self.prepare_db_index()

    def reset_database(self):
        """
        Reset the database to its initial state.
        """
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def _create_repo(self, repo_url_or_path: str) -> None:
        """
        Download and prepare all paths.
        Paths:
        ~/.adalflow/repos/{repo_name} (for url, local path will be the same)
        ~/.adalflow/databases/{repo_name}.pkl
        """
        printc(f"Preparing repo storage for {repo_url_or_path}...")

        try:
            root_path = get_adalflow_default_root_path()

            os.makedirs(root_path, exist_ok=True)
            # url
            if repo_url_or_path.startswith("https://" or "http://"):
                repo_name = repo_url_or_path.split("/")[-1].replace(".git", "")
                save_repo_dir = os.path.join(root_path, "repos", repo_name)
                # download the repo
                download_github_repo(repo_url_or_path, save_repo_dir)
            else:  # local path
                repo_name = os.path.basename(repo_url_or_path)
                save_repo_dir = repo_url_or_path

            save_db_file = os.path.join(root_path, "databases", f"{repo_name}.pkl")
            os.makedirs(save_repo_dir, exist_ok=True)
            os.makedirs(os.path.dirname(save_db_file), exist_ok=True)

            self.repo_paths = {
                "save_repo_dir": save_repo_dir,
                "save_db_file": save_db_file,
            }
            printc(f"Repo paths: {self.repo_paths}")

        except Exception as e:
            printc(f"Failed to create repository structure: {e}")
            raise

    def prepare_db_index(self) -> List[Document]:
        """
        Prepare the indexed database for the repository.
        :return: List of Document objects
        """
        # check the database
        if self.repo_paths and os.path.exists(self.repo_paths["save_db_file"]):
            printc("Loading existing database...")
            self.db = LocalDB.load_state(self.repo_paths["save_db_file"])
            documents = self.db.get_transformed_data(key="split_and_embed")
            if documents:
                return documents

        # prepare the database
        printc("Creating new database...")
        documents = read_all_documents(self.repo_paths["save_repo_dir"])
        self.db = transform_documents_and_save_to_db(
            documents, self.repo_paths["save_db_file"]
        )
        printc(f"total documents: {len(documents)}")
        transformed_docs = self.db.get_transformed_data(key="split_and_embed")
        printc(f"total transformed documents: {len(transformed_docs)}")
        return transformed_docs


if __name__ == "__main__":
    from adalflow.utils import get_logger

    adal.setup_env()

    chat_with_adalflow_lib()
