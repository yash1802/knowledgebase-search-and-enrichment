import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
SQLITE_DB_PATH = DATA_DIR / "metadata.db"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# RAG Configuration
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 250
TOP_K_CHUNKS = 30
CANDIDATE_CHUNKS = 150
SIMILARITY_THRESHOLD = 0.7

# Document-Level Retrieval Configuration
TOP_DOCUMENTS = 3              # Number of documents to retrieve chunks from
DOCUMENT_SCORE_WEIGHT = 0.6    # Weight for max similarity in document scoring
MIN_CHUNKS_PER_DOC = 1         # Minimum chunks to consider a document relevant

# Chunking Strategy Configuration
USE_PAGE_LEVEL_CHUNKING = True          # Enable page-level for PDFs/DOCX
MAX_PAGE_SIZE = 8000                    # Max characters per page chunk
MIN_PAGE_SIZE = 100                     # Min characters per page chunk
SEMANTIC_CHUNK_SIZE = 3000              # For TXT/MD files
SEMANTIC_CHUNK_OVERLAP = 300            # Overlap for semantic chunks

# Cross-Encoder Configuration
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKING = True

# Manual Input Configuration
MANUAL_INFO_FILENAME = "manual_information.txt"

# Chroma Configuration
CHROMA_COLLECTION_NAME = "knowledge_base"

# SQLite Configuration
SQLITE_DB_PATH = str(SQLITE_DB_PATH)

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}
