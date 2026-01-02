from src.rag.retrieval import RetrievalEngine
from src.llm.llm_client import LLMClient


class RAGPipeline:
    """Main RAG pipeline orchestrator."""

    def __init__(self):
        self.retrieval_engine = RetrievalEngine()
        self.llm_client = LLMClient()

    def answer_query(self,query,chat_history = None):   
        
        retrieval_result = self.retrieval_engine.retrieve(query)

        llm_response = self.llm_client.generate_answer(
            query=query,
            context_chunks=retrieval_result.get("chunks", []),
            chat_history=chat_history
        )

        llm_response["query"] = query
        return llm_response
