import openai
from typing import List, Optional
from langchain.embeddings.base import Embeddings  # check version and path!

class EmbeddingService(Embeddings):
    def __init__(self, base_url: str, api_key: str, model_name: str, max_retries: int = 3):
        # Initialize your client
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        # Optionally: determine embedding dimension up front
        self.embedding_dim: Optional[int] = None

    def embed_query(self, text: str) -> List[float]:
        # Note: embed a single text wrapped as list
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text]
        )
        vector = response.data[0].embedding
        # Optionally set embedding_dim if None
        if self.embedding_dim is None:
            self.embedding_dim = len(vector)
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        vectors = [item.embedding for item in response.data]
        if self.embedding_dim is None and vectors:
            self.embedding_dim = len(vectors[0])
        return vectors
