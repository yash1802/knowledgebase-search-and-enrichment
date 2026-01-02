import sqlite3
from datetime import datetime
from config.settings import SQLITE_DB_PATH
import json


class SQLiteStore:
    """Manage SQLite database for document metadata."""

    def __init__(self, db_path = SQLITE_DB_PATH):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                upload_timestamp DATETIME NOT NULL,
                file_type TEXT,
                is_manual_input BOOLEAN DEFAULT 0
            )
        """)

        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                chroma_id TEXT,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)

        # Enrichments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enrichments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT,
                type TEXT NOT NULL,
                content TEXT,
                document_id INTEGER,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)

        # Chats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                last_activity DATETIME NOT NULL
            )
        """)

        # Chat messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                files TEXT,
                metadata TEXT,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)

        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        conn.close()


    def add_document(self,filename,file_path,file_type,is_manual_input = False):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO documents (filename, file_path, upload_timestamp,
                                  file_type, is_manual_input)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, file_path, datetime.now(), file_type, is_manual_input))

        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return doc_id

    def add_chunk(self,document_id,chunk_index,text,chroma_id = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO chunks (document_id, chunk_index, text, chroma_id,
                               created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (document_id, chunk_index, text, chroma_id, datetime.now()))

        chunk_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return chunk_id

    def get_chunk_by_id(self, chunk_id):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT c.*, d.filename, d.file_path
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.id = ?
        """, (chunk_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_chunks_by_ids(self, chunk_ids):
        if not chunk_ids:
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(chunk_ids))
        cursor.execute(f"""
            SELECT c.*, d.filename, d.file_path
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.id IN ({placeholders})
        """, chunk_ids)

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_chunks_by_chroma_ids(self,chroma_ids):
        if not chroma_ids:
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(chroma_ids))
        cursor.execute(f"""
            SELECT c.*, d.filename, d.file_path
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.chroma_id IN ({placeholders})
        """, chroma_ids)

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def add_enrichment(self,query_text,enrichment_type,content = None,document_id = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO enrichments (query_text, type, content, document_id,
                                   created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (query_text, enrichment_type, content, document_id,
              datetime.now()))

        enrichment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return enrichment_id

    def get_all_documents(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM documents ORDER BY upload_timestamp DESC")
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def create_chat(self, name):
        import uuid
        chat_id = f"chat_{uuid.uuid4().hex[:8]}"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO chats (id, name, created_at, last_activity)
            VALUES (?, ?, ?, ?)
        """, (chat_id, name, datetime.now(), datetime.now()))

        conn.commit()
        conn.close()
        return chat_id

    def get_chat(self, chat_id):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_all_chats(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM chats 
            ORDER BY last_activity DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def update_chat_activity(self, chat_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE chats 
            SET last_activity = ? 
            WHERE id = ?
        """, (datetime.now(), chat_id))

        conn.commit()
        conn.close()

    def delete_chat(self, chat_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

        conn.commit()
        conn.close()

    def add_chat_message(self,chat_id,role,content,files = None,metadata = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        files_json = json.dumps(files) if files else None
        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute("""
            INSERT INTO chat_messages 
            (chat_id, role, content, files, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (chat_id, role, content, files_json, metadata_json, datetime.now()))

        message_id = cursor.lastrowid

        cursor.execute("""
            UPDATE chats 
            SET last_activity = ? 
            WHERE id = ?
        """, (datetime.now(), chat_id))

        conn.commit()
        conn.close()
        return message_id

    def get_chat_messages(self, chat_id):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM chat_messages 
            WHERE chat_id = ? 
            ORDER BY timestamp ASC
        """, (chat_id,))

        rows = cursor.fetchall()
        conn.close()

        messages = []
        for row in rows:
            msg = dict(row)
            # Parse JSON fields
            if msg.get("files"):
                msg["files"] = json.loads(msg["files"])
            if msg.get("metadata"):
                msg["metadata"] = json.loads(msg["metadata"])
            messages.append(msg)

        return messages

    def clear_chat_history(self, chat_id):
        """
        Delete all messages for a chat, keep chat.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM chat_messages WHERE chat_id = ?
        """, (chat_id,))

        conn.commit()
        conn.close()

    def get_chat_count(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM chats")
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def get_default_chat_id(self):
        """
        Get the first/default chat ID.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id FROM chats 
            ORDER BY created_at ASC 
            LIMIT 1
        """)
        result = cursor.fetchone()

        conn.close()
        return result[0] if result else None

    def get_chunks_by_document_id(self,document_id):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT c.*, d.filename, d.file_path
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = ?
        """, (document_id,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def delete_document(self, document_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete chunks
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        # Delete document
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

        conn.commit()
        conn.close()
    
    def add_feedback(self, message_id, rating, comment=None):
        """
        Add user feedback for a message.
        rating: 1 for thumbs up, -1 for thumbs down
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO feedback (message_id, rating, comment, timestamp)
            VALUES (?, ?, ?, ?)
        """, (message_id, rating, comment, datetime.now()))

        conn.commit()
        conn.close()

    def get_message_feedback(self, message_id):
        """
        Check if a message already has feedback.
        Returns tuple (rating, comment) or None.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT rating, comment FROM feedback WHERE message_id = ?", (message_id,))
        row = cursor.fetchone()
        conn.close()
        return row