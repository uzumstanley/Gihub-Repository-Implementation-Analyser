# Repository Architecture

This document explains how the different components of the RAG (Retrieval-Augmented Generation) system work together.

## File Structure and Dependencies

```mermaid
graph TD
    config[config.py] --> rag[rag.py]
    config --> data_pipeline[data_pipeline.py]
    data_pipeline --> test_rag[test_rag.py]
    data_pipeline --> app_repo[app_repo.py]
    rag --> app[app.py]
    rag --> app_repo
    test_rag --> app
```

## Data Flow

```mermaid
flowchart TD
    subgraph Input
        A[User Query] --> B[Streamlit Interface]
        C[Repository/Documents] --> D[Document Processor]
    end

    subgraph Processing
        B --> E[RAG System]
        D --> F[Text Splitter]
        F --> G[Embedder]
        G --> H[FAISS Index]
        H --> E
    end

    subgraph Output
        E --> I[Response]
        E --> J[Context]
        I --> K[Chat Interface]
        J --> K
    end
```

## Component Responsibilities

### Core Components

1. `data_pipeline.py`
```mermaid
flowchart LR
    A[Input Documents] --> B[read_all_documents]
    B --> C[Text Splitter]
    C --> D[Embedder]
    D --> E[Database]
    
    subgraph Functions
        B
        F[download_github_repo]
        G[extract_class_definition]
        H[transform_documents]
    end
```

2. `rag.py`
```mermaid
flowchart LR
    A[User Query] --> B[RAG Component]
    B --> C[FAISS Retriever]
    B --> D[Generator]
    B --> E[Memory]
    
    subgraph Components
        C --> F[Retrieved Docs]
        D --> G[Generated Response]
        E --> H[Conversation History]
    end
```

### Interface Components

1. `app.py` (Demo Interface)
```mermaid
flowchart LR
    A[Sample Data] --> B[Test Database]
    B --> C[RAG System]
    C --> D[Chat Interface]
```

2. `app_repo.py` (Repository Analysis Interface)
```mermaid
flowchart LR
    A[Repository Path] --> B[Document Processor]
    B --> C[Database]
    C --> D[RAG System]
    D --> E[Chat Interface]
```

## Configuration (`config.py`)

```mermaid
flowchart TD
    subgraph Configuration
        A[Text Splitter Config]
        B[Embedder Config]
        C[Retriever Config]
        D[Generator Config]
    end

    subgraph Parameters
        A --> A1[chunk_size]
        A --> A2[chunk_overlap]
        B --> B1[model]
        B --> B2[dimensions]
        C --> C1[top_k]
        D --> D1[temperature]
    end
```

## Data Processing Pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant I as Interface
    participant D as Data Pipeline
    participant R as RAG System
    participant DB as Database

    U->>I: Input Query/Load Repo
    I->>D: Process Documents
    D->>D: Split Text
    D->>D: Generate Embeddings
    D->>DB: Store Documents
    I->>R: Send Query
    R->>DB: Retrieve Context
    R->>R: Generate Response
    R->>I: Return Response
    I->>U: Display Results
```

## Memory Management

```mermaid
flowchart LR
    A[User Query] --> B[Memory Component]
    B --> C[Current Conversation]
    C --> D[Dialog Turns]
    D --> E[Context for Next Query]
```

This architecture provides:
- Modular components that can be easily modified
- Clear separation of concerns
- Two interfaces (demo and repository analysis)
- Configurable parameters for text processing
- Conversation memory management