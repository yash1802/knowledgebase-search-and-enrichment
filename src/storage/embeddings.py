from openai import OpenAI
from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL


class EmbeddingGenerator:
    """Generate embeddings for text using OpenAI."""

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = EMBEDDING_MODEL

    def generate_embedding(self, text):
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError(f"Error generating embedding: {str(e)}")

    def generate_embeddings_batch(self, texts):
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {str(e)}")

