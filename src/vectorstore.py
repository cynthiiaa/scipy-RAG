"""
Vector Store Wrapper

High-level wrapper around ChromaDB for storing and retrieving document embeddings.

Workshop goals:
- Keep the interface simple
- Make score semantics explicit (and stable for filters/sorting)
- Avoid overly-broad exception handling where it matters
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from embeddings import EmbeddingProvider


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""
    id: str
    text: str
    metadata: dict
    distance: float
    score: float  # Normalized similarity score in [0, 1]


class VectorStore:
    """
    A small wrapper around ChromaDB.

    Supports two modes:
    - Chroma-managed embeddings (default): uses OpenAIEmbeddingFunction inside Chroma
    - Manual embeddings: pass an EmbeddingProvider, and VectorStore will embed before add/query
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.collection_name = collection_name

        # Client
        if persist_directory:
            self.persist_directory = Path(persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        else:
            self.persist_directory = None
            self.client = chromadb.Client()

        # Embeddings
        self.embedding_provider = embedding_provider
        if embedding_provider is None:
            # Chroma-managed embedding function
            self._embedding_function = OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=embedding_model
            )
        else:
            self._embedding_function = None

        self._init_collection()

    def _init_collection(self) -> None:
        """Get or create the collection."""
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function,
            )
        except Exception:
            # Collection doesn't exist (or embedding function mismatch); create it.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

    def reset_collection(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._init_collection()

    def add_documents(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
        embeddings: list[list[float]] | None = None
    ) -> None:
        if not texts:
            return

        if len(texts) != len(ids):
            raise ValueError("texts and ids must have the same length")

        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas must be None or the same length as texts")

        # Manual embedding mode
        if self.embedding_provider is not None and embeddings is None:
            embeddings = self.embedding_provider.embed_batch(texts)

        add_kwargs: dict = {"ids": ids, "documents": texts}
        if metadatas is not None:
            add_kwargs["metadatas"] = metadatas
        if embeddings is not None:
            add_kwargs["embeddings"] = embeddings

        self.collection.add(**add_kwargs)

    def add_document(
        self,
        text: str,
        id: str,
        metadata: dict | None = None,
        embedding: list[float] | None = None
    ) -> None:
        self.add_documents(
            texts=[text],
            ids=[id],
            metadatas=[metadata] if metadata is not None else None,
            embeddings=[embedding] if embedding is not None else None,
        )

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """
        Convert Chroma cosine distance -> normalized similarity score in [0, 1].

        Chroma uses cosine distance: d = 1 - cos_sim, where cos_sim ∈ [-1, 1].
        So:
          cos_sim = 1 - d  ∈ [-1, 1]
          score   = (cos_sim + 1) / 2  ∈ [0, 1]
        """
        cos_sim = 1.0 - float(distance)
        score = (cos_sim + 1.0) / 2.0
        # Clamp for safety
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
        where_document: dict | None = None,
        include_embeddings: bool = False
    ) -> list[SearchResult]:
        # Query embedding (manual mode)
        query_embedding: list[float] | None = None
        if self.embedding_provider is not None:
            query_embedding = self.embedding_provider.embed(query)

        include_fields = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include_fields.append("embeddings")

        query_kwargs: dict = {"n_results": n_results, "include": include_fields}

        if query_embedding is not None:
            query_kwargs["query_embeddings"] = [query_embedding]
        else:
            query_kwargs["query_texts"] = [query]

        if where is not None:
            query_kwargs["where"] = where
        if where_document is not None:
            query_kwargs["where_document"] = where_document

        results = self.collection.query(**query_kwargs)

        # Convert to SearchResult objects
        search_results: list[SearchResult] = []
        ids0 = results.get("ids", [[]])[0] or []
        dists0 = results.get("distances", [[]])[0] or []
        docs0 = results.get("documents", [[]])[0] or []
        metas0 = results.get("metadatas", [[]])[0] if results.get("metadatas") else [{}] * len(ids0)

        for i in range(len(ids0)):
            distance = float(dists0[i]) if i < len(dists0) else 0.0
            search_results.append(SearchResult(
                id=ids0[i],
                text=docs0[i] if i < len(docs0) else "",
                metadata=metas0[i] if i < len(metas0) else {},
                distance=distance,
                score=self._distance_to_score(distance)
            ))

        return search_results

    def get_by_id(self, id: str) -> Optional[SearchResult]:
        results = self.collection.get(ids=[id], include=["documents", "metadatas"])
        ids = results.get("ids") or []
        if not ids:
            return None

        doc_list = results.get("documents") or [""]
        meta_list = results.get("metadatas") or [{}]

        return SearchResult(
            id=ids[0],
            text=doc_list[0] if doc_list else "",
            metadata=meta_list[0] if meta_list else {},
            distance=0.0,
            score=1.0
        )

    def get_by_metadata(self, where: dict) -> list[SearchResult]:
        results = self.collection.get(where=where, include=["documents", "metadatas"])
        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or [{}] * len(ids)

        out: list[SearchResult] = []
        for i, doc_id in enumerate(ids):
            out.append(SearchResult(
                id=doc_id,
                text=docs[i] if i < len(docs) else "",
                metadata=metas[i] if i < len(metas) else {},
                distance=0.0,
                score=1.0
            ))
        return out

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)

    def count(self) -> int:
        return int(self.collection.count())

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.list_collections()]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing VectorStore...")

    store = VectorStore(collection_name="test_collection")
    store.reset_collection()

    docs = [
        ("scipy.optimize.minimize minimizes a function", "doc1", {"module": "optimize"}),
        ("scipy.integrate.quad computes integrals", "doc2", {"module": "integrate"}),
        ("scipy.linalg.solve solves linear systems", "doc3", {"module": "linalg"}),
    ]

    for text, doc_id, meta in docs:
        store.add_document(text, doc_id, meta)

    print(f"Added {store.count()} documents")

    results = store.search("how to minimize a function", n_results=2)
    print("\nSearch results for 'how to minimize a function':")
    for r in results:
        print(f"  [{r.score:.3f}] {r.id}: {r.text[:50]}...")

    results = store.search("compute", n_results=2, where={"module": "integrate"})
    print("\nSearch with filter (module=integrate):")
    for r in results:
        print(f"  [{r.score:.3f}] {r.id}: {r.text[:50]}...")
