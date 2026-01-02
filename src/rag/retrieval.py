from src.storage.embeddings import EmbeddingGenerator
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_store import SQLiteStore
from config.settings import (
    TOP_K_CHUNKS,
    SIMILARITY_THRESHOLD,
    CANDIDATE_CHUNKS,
    TOP_DOCUMENTS,
    CROSS_ENCODER_MODEL,
    DOCUMENT_SCORE_WEIGHT,
    MIN_CHUNKS_PER_DOC
)
from sentence_transformers import CrossEncoder


class RetrievalEngine:
    """Handle query retrieval from vector store."""

    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.chroma_store = ChromaStore()
        self.sqlite_store = SQLiteStore()
        self.top_k = TOP_K_CHUNKS
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.model = CrossEncoder(CROSS_ENCODER_MODEL)
        self.score_weight = DOCUMENT_SCORE_WEIGHT
        self.min_chunks = MIN_CHUNKS_PER_DOC

    
    def rerank(self, query, chunks, top_k = 30):
        """
        Re-rank chunks using cross-encoder.
        """
        if not chunks:
            return []

        pairs = [(query, chunk.get('text', '')) for chunk in chunks]

        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk['rerank_score'] = float(score)
            chunk['original_similarity'] = chunk.get('similarity', 0.0)

        reranked = sorted(
            chunks,
            key=lambda x: x.get('rerank_score', 0.0),
            reverse=True
        )

        return reranked[:top_k]
    
    
    def _score_documents(self, query_embedding, top_k_candidates=100):
        results = self.chroma_store.search(query_embedding, top_k=top_k_candidates)
        
        if not results:
            return {}
        
        # Group chunks by document and aggregate scores
        doc_scores = {}
        for result in results:
            doc_id = int(result['metadata'].get('document_id'))
            similarity = result['similarity']
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'max_sim': similarity,
                    'total_sim': similarity,
                    'count': 1,
                    'similarities': [similarity]
                }
            else:
                doc_scores[doc_id]['max_sim'] = max(
                    doc_scores[doc_id]['max_sim'],
                    similarity
                )
                doc_scores[doc_id]['total_sim'] += similarity
                doc_scores[doc_id]['count'] += 1
                doc_scores[doc_id]['similarities'].append(similarity)
        
        final_scores = {}
        filtered_docs = []
        for doc_id, scores in doc_scores.items():
            if scores['count'] < self.min_chunks:
                filtered_docs.append(doc_id)
                continue
            
            avg_sim = scores['total_sim'] / scores['count']
            max_sim = scores['max_sim']
            
            final_scores[doc_id] = (
                max_sim * self.score_weight +
                avg_sim * (1 - self.score_weight)
            )
        
        return final_scores
    
    
    def _get_top_documents(self, doc_scores, top_n=3):
        if not doc_scores:
            return []
        
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [doc_id for doc_id, score in sorted_docs[:top_n]]
    
    
    def retrieve(self,query):
        """
        Document-level retrieval with two-stage re-ranking.
        """
        # Embed query
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Score documents (get top documents by relevance)
        doc_scores = self._score_documents(
            query_embedding,
            top_k_candidates=CANDIDATE_CHUNKS
        )

        if not doc_scores:
            return {
                "chunks": [],
                "max_similarity": 0.0,
                "num_chunks": 0,
                "retrieval_quality": "none",
                "top_documents": []
            }

        # Get top N documents
        top_doc_ids = self._get_top_documents(
            doc_scores,
            top_n=TOP_DOCUMENTS
        )

        # Retrieve ALL chunks from top documents
        all_chunks = []
        for doc_id in top_doc_ids:
            doc_chunks = self.sqlite_store.get_chunks_by_document_id(doc_id)
            # Add document relevance score to each chunk
            for chunk in doc_chunks:
                chunk['document_score'] = doc_scores[doc_id]
                chunk['document_id'] = doc_id
            all_chunks.extend(doc_chunks)

        if not all_chunks:
            return {
                "chunks": [],
                "max_similarity": 0.0,
                "num_chunks": 0,
                "retrieval_quality": "none",
                "top_documents": top_doc_ids
            }

        # Re-rank with cross-encoder
        final_chunks = self.rerank(query, all_chunks, self.top_k)
        
        # Calculate max similarity for assessment
        max_similarity = max([c.get('document_score', 0) for c in final_chunks])

        return {
            "chunks": final_chunks,
            "max_similarity": max_similarity,
            "num_chunks": len(final_chunks),
            "top_documents": top_doc_ids
        }
