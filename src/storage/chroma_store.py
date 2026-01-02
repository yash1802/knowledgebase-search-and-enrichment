import chromadb
from chromadb.config import Settings
from config.settings import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME


class ChromaStore:
    """Manage Chroma vector database."""

    def __init__(self, collection_name = CHROMA_COLLECTION_NAME):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            collection = self.client.get_collection(name=self.collection_name)
            return collection
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name, 
                metadata={"hnsw:space": "cosine"}
            )
            return collection

    def add_chunks(self,chunks,embeddings,metadata):
        ids = [meta.get("chroma_id", f"chunk_{i}") for i, meta in
               enumerate(metadata)]

        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata,
            ids=ids
        )

    def search(self,query_embedding,top_k = 5):
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1 - results["distances"][0][i],
                    "metadata": results["metadatas"][0][i] if
                    results["metadatas"] else {}
                })

        return formatted_results

    def delete_chunks_by_ids(self, chunk_ids):
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

