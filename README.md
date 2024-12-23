# RAG Code Assistant

A Retrieval-Augmented Generation (RAG) system for analyzing and understanding code repositories. The system provides both a command-line interface and a web UI for interacting with your codebase.

## Features

- Code-aware responses using RAG
- Memory for maintaining conversation context
- Support for multiple programming languages
- Interactive web interface
- Command-line interface

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

### Web Interface

1. Run the demo version (with test data):
```bash
poetry run streamlit run app.py
```

2. Run the repository analysis version:
```bash
poetry run streamlit run app_repo.py
```

### Command Line Interface

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