import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL


class EmbeddingGenerator:
    """Generate embeddings for text using OpenAI."""

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = EMBEDDING_MODEL

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError))
    )
    def _create_embedding(self, **kwargs):
        return self.client.embeddings.create(**kwargs)

    def generate_embedding(self, text):
        try:
            response = self._create_embedding(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError(f"Error generating embedding: {str(e)}")

    def generate_embeddings_batch(self, texts):
        try:
            response = self._create_embedding(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {str(e)}")