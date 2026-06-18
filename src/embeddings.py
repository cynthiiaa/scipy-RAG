"""
Embedding Providers

This module provides a unified interface for generating embeddings
using different providers (OpenAI, Ollama).
"""

import os
import time
from abc import ABC, abstractmethod
from openai import OpenAI


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
    """
    OpenAI embedding provider.

    Uses OpenAI's embedding API for high-quality embeddings.
    """

    # Embedding dimensions for different models
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,  # legacy
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None
    ):
        """
        Initialize OpenAI embeddings.

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._dimension = self.DIMENSIONS.get(model, 1536)

    def _with_retries(self, fn, max_retries: int = 3):
        """
        Execute a callable with simple exponential backoff.

        Keeps workshop runs smooth on transient 429/5xx/network errors.
        """
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception:
                # Best-effort retry. Keep it simple for workshop material.
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt))

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self._with_retries(lambda: self.client.embeddings.create(
            input=text,
            model=self.model
        ))
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        More efficient than calling embed() multiple times.
        """
        if not texts:
            return []

        response = self._with_retries(lambda: self.client.embeddings.create(
            input=texts,
            model=self.model
        ))
        # Sort by index to maintain order
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    @property
    def dimension(self) -> int:
        return self._dimension


class OllamaEmbeddings(EmbeddingProvider):
    """
    Ollama embedding provider.

    Uses locally-running Ollama for embeddings. Good for offline use
    and when you want to avoid API costs.
    """

    # Approximate dimensions for common models
    DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = None
    ):
        """
        Initialize Ollama embeddings.

        Args:
            model: Ollama model name
            host: Ollama host URL (uses OLLAMA_HOST env var or default)
        """
        import ollama

        self.model = model
        self._ollama = ollama

        # Set host if provided
        if host:
            os.environ["OLLAMA_HOST"] = host

        self._dimension = self.DIMENSIONS.get(model, 768)

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self._ollama.embeddings(
            model=self.model,
            prompt=text
        )
        return response.get("embedding")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Note: Ollama doesn't have native batch support, so this
        processes texts sequentially.
        """
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_provider(
    provider: str = "openai",
    model: str = None,
    **kwargs
) -> EmbeddingProvider:
    """
    Factory function to get an embedding provider.

    Args:
        provider: Provider name ('openai' or 'ollama')
        model: Model name (provider-specific)
        **kwargs: Additional provider-specific arguments

    Returns:
        EmbeddingProvider instance
    """
    if provider == "openai":
        model = model or "text-embedding-3-small"
        return OpenAIEmbeddings(model=model, **kwargs)
    elif provider == "ollama":
        model = model or "nomic-embed-text"
        return OllamaEmbeddings(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# Demonstration
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing OpenAI Embeddings...")
    openai_emb = OpenAIEmbeddings()
    text = "SciPy is a Python library for scientific computing"
    embedding = openai_emb.embed(text)
    print(f"  Text: {text}")
    print(f"  Embedding dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")

    print("\nTesting batch embedding...")
    texts = [
        "scipy.optimize.minimize",
        "scipy.integrate.quad",
        "scipy.linalg.solve"
    ]
    embeddings = openai_emb.embed_batch(texts)
    print(f"  Generated {len(embeddings)} embeddings")

    print("\nTesting Ollama Embeddings...")
    try:
        ollama_emb = OllamaEmbeddings()
        embedding = ollama_emb.embed(text)
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"  Ollama not available: {e}")
