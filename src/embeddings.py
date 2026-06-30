"""
Embedding Providers

This module provides an interface for generating embeddings using different providers (OpenAI, Ollama).
"""

import os
import time
from abc import ABC, abstractmethod
from openai import OpenAI

from errors import require_openai_key, import_ollama, handle_ollama_error


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider."""

    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.model = model
        api_key = require_openai_key(api_key, for_embeddings=True)
        self.client = OpenAI(api_key=api_key)
        self._dimension = self.DIMENSIONS.get(model, 1536)

    def _with_retries(self, fn, max_retries: int = 3):
        """Execute with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt))

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self._with_retries(lambda: self.client.embeddings.create(
            input=text, model=self.model
        ))
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        response = self._with_retries(lambda: self.client.embeddings.create(
            input=texts, model=self.model
        ))
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    @property
    def dimension(self) -> int:
        return self._dimension


class OllamaEmbeddings(EmbeddingProvider):
    """Ollama embedding provider for local embeddings."""

    DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(self, model: str = "nomic-embed-text", host: str = None):
        self.model = model
        self._ollama = import_ollama(model)

        if host:
            os.environ["OLLAMA_HOST"] = host

        self._dimension = self.DIMENSIONS.get(model, 768)

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        try:
            response = self._ollama.embeddings(model=self.model, prompt=text)
            return response.get("embedding")
        except Exception as e:
            handle_ollama_error(e, self.model)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (sequential)."""
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_provider(provider: str = "openai", model: str = None, **kwargs) -> EmbeddingProvider:
    """Factory function to get an embedding provider."""
    if provider == "openai":
        return OpenAIEmbeddings(model=model or "text-embedding-3-small", **kwargs)
    elif provider == "ollama":
        return OllamaEmbeddings(model=model or "nomic-embed-text", **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing OpenAI Embeddings...")
    openai_emb = OpenAIEmbeddings()
    text = "SciPy is a Python library for scientific computing"
    embedding = openai_emb.embed(text)
    print(f"  Dimension: {len(embedding)}, First 5: {embedding[:5]}")

    print("\nTesting Ollama Embeddings...")
    try:
        ollama_emb = OllamaEmbeddings()
        embedding = ollama_emb.embed(text)
        print(f"  Dimension: {len(embedding)}, First 5: {embedding[:5]}")
    except Exception as e:
        print(f"  Ollama not available: {e}")
