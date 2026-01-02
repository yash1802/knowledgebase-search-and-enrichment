from datetime import datetime
from pathlib import Path
import sqlite3
from src.ingestion.document_processor import DocumentProcessor
from src.storage.embeddings import EmbeddingGenerator
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_store import SQLiteStore
from config.settings import UPLOADS_DIR, MANUAL_INFO_FILENAME


class ManualInputProcessor:
    """Process manually entered information as documents."""

    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.chroma_store = ChromaStore()
        self.sqlite_store = SQLiteStore()

    def process_manual_input(self,user_input,query_text = None):
        """
        Process manual input and add incrementally to consolidated file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = user_input.strip()
        
        file_path = UPLOADS_DIR / MANUAL_INFO_FILENAME
        file_exists = file_path.exists()
        
        if file_exists:
            doc_id = self._get_manual_info_document_id()
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}]\n{content}\n\n")
            
            # Get next chunk index
            next_chunk_index = self._get_next_chunk_index(doc_id)
            
            # Exclude timestamp and embed ONLY the actual information text
            embedding = self.embedding_generator.generate_embedding(content)
            
            # Add new chunk to SQLite
            chroma_id = f"doc_{doc_id}_chunk_{next_chunk_index}"
            chunk_id = self.sqlite_store.add_chunk(
                document_id=doc_id,
                chunk_index=next_chunk_index,
                text=content,
                chroma_id=chroma_id
            )
            
            # Add new chunk to Chroma
            self.chroma_store.add_chunks(
                chunks=[content],
                embeddings=[embedding],
                metadata=[{
                    "document_id": str(doc_id),
                    "chunk_id": str(chunk_id),
                    "chunk_index": str(next_chunk_index),
                    "filename": MANUAL_INFO_FILENAME,
                    "chroma_id": chroma_id
                }]
            )
            
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"[{timestamp}]\n{content}\n\n")
            
            # Add to SQLite
            doc_id = self.sqlite_store.add_document(
                filename=MANUAL_INFO_FILENAME,
                file_path=str(file_path),
                file_type=".txt",
                is_manual_input=True
            )
            
            embedding = self.embedding_generator.generate_embedding(content)
            
            chroma_id = f"doc_{doc_id}_chunk_0"
            chunk_id = self.sqlite_store.add_chunk(
                document_id=doc_id,
                chunk_index=0,
                text=content,
                chroma_id=chroma_id
            )
            
            # Add chunk to Chroma
            self.chroma_store.add_chunks(
                chunks=[content],
                embeddings=[embedding],
                metadata=[{
                    "document_id": str(doc_id),
                    "chunk_id": str(chunk_id),
                    "chunk_index": "0",
                    "filename": MANUAL_INFO_FILENAME,
                    "chroma_id": chroma_id
                }]
            )
        
        # Record enrichment
        self.sqlite_store.add_enrichment(
            query_text=query_text,
            enrichment_type="manual",
            content=user_input,
            document_id=doc_id
        )
        
        return {
            "success": True,
            "document_id": doc_id,
            "filename": MANUAL_INFO_FILENAME,
            "message": f"Added manual input to {MANUAL_INFO_FILENAME}"
        }

    def _get_manual_info_document_id(self):
        """Get the document_id for manual_information.txt from SQLite."""
        conn = sqlite3.connect(self.sqlite_store.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id FROM documents 
            WHERE filename = ? AND is_manual_input = 1
        """, (MANUAL_INFO_FILENAME,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            raise ValueError(f"{MANUAL_INFO_FILENAME} not found in database")

    def _get_next_chunk_index(self, doc_id):
        conn = sqlite3.connect(self.sqlite_store.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MAX(chunk_index) FROM chunks 
            WHERE document_id = ?
        """, (doc_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] is not None:
            return result[0] + 1
        
        return 0

