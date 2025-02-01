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

- [ ] Conditional retrieval. Sometimes users just want to clarify a past conversation, no extra context needed.
- [ ] Create an evaluation dataset  
- [ ] Evaluate the RAG performance on the dataset  
- [ ] Auto-optimize the RAG model
