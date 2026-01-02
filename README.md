# Knowledge Base Search & Enrichment System 



## Overview 


A Retrieval-Augmented Generation (RAG) application that allows users to upload documents (PDF, DOCX, TXT, MD), manage a knowledge base and perform semantic search. 

The application is capable of detecting uncertainty and lack of information as well as suggesting ways to address said lack. 

----

## Directory Structure

knowledgebase-search-and-enrichment/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── config/
│   └── settings.py             # Global configuration (paths, model names, constants)
└── src/
    ├── ingestion/
    │   ├── document_processor.py # Logic for parsing, cleaning, and chunking files
    │   └── manual_input.py       # Handling direct text input/enrichment
    ├── llm/
    │   ├── llm_client.py         # OpenAI API wrapper, retry logic, and response parsing
    │   └── prompts.py            # Centralized prompt templates
    ├── rag/
    │   ├── rag_pipeline.py       # Orchestrator connecting retrieval and generation
    │   └── retrieval.py          # Hybrid retrieval logic (Vector Search + Re-ranking)
    └── storage/
        ├── chroma_store.py       # Vector database (ChromaDB) interface
        ├── sqlite_store.py       # Relational database (SQLite) for metadata/history
        └── embeddings.py         # Embedding generation service


----

## Implementation details

### 1. Chunking Logic - content-aware chunking strategies based on file type.

- PDFs & DOCX: Uses a page-level strategy combined with sliding windows. If a page exceeds MAX_PAGE_SIZE (8000 chars), it is recursively split. Small pages are merged to preserve context.

- Markdown: Chunks by headers (Sections #, ##), ensuring that semantic sections remain intact.

- Plain Text: Uses a standard sliding window approach (default 1000 chars) with overlap (200 chars) to maintain continuity across boundaries.



### 2. Retrieval Logic - implements a two-stage retrieval process.

- Candidate Generation: Retrieves a broad set of candidates from ChromaDB using Cosine Similarity.

- Document Scoring: Aggregates chunk scores to identify the most relevant documents overall, preventing a single lucky chunk from dominating the context. 

- Re-Ranking: The top results are passed through a Cross-Encoder which reads the query and chunk together to output a precise relevance score leading to higher accuracy over simple vector similarity.



### 3. LLM Logic.

- Intent detection: A classification step determines whether the user is asking a question, providing new information or is simply making conversation and makes the appropriate call to the LLM. 

- Model: Uses gpt-4o-mini for a balance of speed, cost and intelligence. 

- Determinism: Sets temperature=0.0 to ensure consistent, reproducible outputs.



### 4. Prompt Engineering. 

- Structured Output: Forces the model to return valid JSON containing specific keys: answer, confidence, missing_info and sources. 

- Few-Shot Prompting: The system prompt includes distinct examples (e.g., "Explicit evidence," "Absence of evidence," "Partial info") to teach the model how to calibrate confidence scores for common and edge cases.

- Context Isolation: Retrieved documents are wrapped in XML tags (<documents>) to clearly delineate external data from system instructions. 


----


## Trade-offs

1. Database Selection

- In production, PostGres + pgvector or a managed vector DB such as Pinecone, Weaviate are used due to superior indexing, filtering and scalability. However, using them would have incured operational overhead (provisioning, docker, etc). 
- Given the time constraint and the scope of the task, using SQLite + Chroma is sufficient whilst being trivial to set up. 

2. Frontend Framework. 

- Considering that the backend and AI focused nature of the assignment, opted for Streamlit instead of React/JS to avoid sinking time into frontend tooling, routing and state management.

3. Model Selection.

- Using open-source models locally has advantages like privacy as well as no API management and are well suited for this project especially since the knowledge base itself is local. However, using them would require quantization and other setup which would have cost time. 
- Hence, GPT‑4o‑mini was used instead which has strong instruction following, supports structured outputs, is multimodal and is relatively inexpensive.


----


## Setup and Running

1. Clone and navigate to the project:

```

cd Knowledgebase_search_and_enrichment

```



2. Install dependencies:

```

pip install -r requirements.txt

```



3. Set up environment variables:

Add OPENAI's API Key to .env file. 

```

OPENAI_API_KEY=your_openai_api_key_here

```



4. Run the application.

```

streamlit run app.py

```

----