# GithubChat

A RAG assistant to allow you to chat with any github repo. 
Learn fast. The default repo is AdalFlow github repo.

## Setup

1. Install dependencies:
```bash
poetry install
```

2. Set up OpenAI API key:

Create a `.streamlit/secrets.toml` file in your project root:
```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
```

Add your OpenAI API key to `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

## Running the Application

Run the streamlit app:
```bash
poetry run streamlit run app.py
```

## ROADMAP
- [x] Clearyly structured RAG that can prepare a repo, persit from reloading, and answer questions.
  - `DatabaseManager` in `src/data_pipeline.py` to manage the database.
  - `RAG` class in `src/rag.py` to manage the whole RAG lifecycle.

<!-- CREATE Checklist -->
- [ ] Create an evaluation dataset  
- [ ] Evaluate the RAG performance on the dataset  
- [ ] Auto-optimize the RAG model
<!-- ## Learn

## Local Storage
We use adalflow's root directory, which is at ~/.adalflow.
- repos/repo_name/...
- repos/repo_name_db/...

- data_pipeline.py: From the main and local code test, you will know the process of download repo and chunk files, and embed the chunks.
- rag.py: The main code of the RAG model. -->

<!-- ### Command Line Interface

Run the RAG system directly:
```bash
poetry run python rag.py
```

## Usage Examples

1. **Demo Version (app.py)**
   - Ask about Alice (software engineer)
   - Ask about Bob (data scientist)
   - Ask about the company cafeteria
   - Test memory with follow-up questions

2. **Repository Analysis (app_repo.py)**
   - Enter your repository path
   - Click "Load Repository"
   - Ask questions about classes, functions, or code structure
   - View implementation details in expandable sections

## Security Note

- Never commit your `.streamlit/secrets.toml` file
- Add it to your `.gitignore`
- Keep your API key secure

## Example Queries

- "What does the RAG class do?"
- "Show me the implementation of the Memory class"
- "How is data processing handled?"
- "Explain the initialization process"

## TODO

- [ ] Add evaluation metrics
- [ ] Improve the embedding model
- [ ] Improve the text splitter and chunking
- [ ] Improve the retriever -->