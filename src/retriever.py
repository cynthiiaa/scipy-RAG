"""
Retrieval Module

This module provides retrieval logic for the RAG system,
including query preprocessing, multi-query retrieval, and
result ranking.
"""

from dataclasses import dataclass
from typing import Optional

from vectorstore import VectorStore, SearchResult


@dataclass
class RetrievalResult:
    """Represents the result of a retrieval operation."""
    query: str
    results: list[SearchResult]
    total_found: int
    filters_applied: dict


class Retriever:
    """
    Retriever for the RAG system.

    Handles query preprocessing, retrieval, and result ranking.

    Note: VectorStore.score is expected to be a normalized similarity in [0, 1], where higher is better.

    """

    def __init__(
        self,
        vector_store: VectorStore,
        default_top_k: int = 5,
        score_threshold: float = 0.5
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: VectorStore instance
            default_top_k: Default number of results to return
            score_threshold: Minimum similarity score to include
        """
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_module: str = None,
        filter_type: str = None,
        chunk_types: list[str] = None,
        min_score: float = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results (uses default if None)
            filter_module: Filter by module (e.g., 'scipy.optimize')
            filter_type: Filter by doc type ('function', 'class', etc.)
            chunk_types: Filter by chunk types ('summary', 'examples', etc.)
            min_score: Minimum similarity score

        Returns:
            RetrievalResult with matching documents
        """
        top_k = self.default_top_k if top_k is None else top_k
        min_score = self.score_threshold if min_score is None else min_score

        # Build metadata filter
        where = {}
        if filter_module:
            where["module"] = filter_module
        if filter_type:
            where["doc_type"] = filter_type
        if chunk_types and len(chunk_types) == 1:
            where["chunk_type"] = chunk_types[0]

        # Search vector store
        results = self.vector_store.search(
            query=query,
            n_results=top_k * 2,  # Get more to filter
            where=where if where else None
        )

        # Filter by score
        # Note: Assumes higher score is more relevant (similarity), not distance.

        filtered_results = [r for r in results if r.score >= min_score]

        # Filter by chunk types if multiple specified
        if chunk_types and len(chunk_types) > 1:
            filtered_results = [
                r for r in filtered_results
                if r.metadata.get('chunk_type') in chunk_types
            ]

        # Limit to top_k
        filtered_results = filtered_results[:top_k]

        return RetrievalResult(
            query=query,
            results=filtered_results,
            total_found=len(filtered_results),
            filters_applied={
                "module": filter_module,
                "doc_type": filter_type,
                "chunk_types": chunk_types,
                "min_score": min_score
            }
        )

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 3,
        include_examples: bool = True,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve with smart context selection.

        Prioritizes summary chunks and optionally includes examples.

        Args:
            query: Search query
            top_k: Number of unique functions to retrieve
            include_examples: Whether to include example chunks
            **kwargs: Additional filter arguments

        Returns:
            RetrievalResult with context-aware results
        """
        # First, get summary chunks for the top functions
        summary_result = self.retrieve(
            query=query,
            top_k=top_k,
            chunk_types=["summary"],
            **kwargs
        )

        results = list(summary_result.results)
        seen_functions = {r.metadata.get('function_name') for r in results if r.metadata.get('function_name')}

        # Add example chunks for the found functions
        if include_examples:
            for func_name in seen_functions:
                example_results = self.vector_store.get_by_metadata(
                    where={
                        "$and": [
                            {"function_name": func_name},
                            {"chunk_type": "examples"}
                        ]
                    }
                )
                results.extend(example_results)

        return RetrievalResult(
            query=query,
            results=results,
            total_found=len(results),
            filters_applied={"include_examples": include_examples, **kwargs}
        )

    def retrieve_multi_query(
        self,
        queries: list[str],
        top_k_per_query: int = 3,
        deduplicate: bool = True,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve using multiple query variations.

        Useful for improving recall by querying from different angles.

        Args:
            queries: List of query variations
            top_k_per_query: Results per query
            deduplicate: Remove duplicate results
            **kwargs: Additional filter arguments

        Returns:
            Combined RetrievalResult
        """
        all_results = []
        seen_ids = set()

        for query in queries:
            result = self.retrieve(
                query=query,
                top_k=top_k_per_query,
                **kwargs
            )

            for r in result.results:
                if not deduplicate or r.id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.id)

        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)

        return RetrievalResult(
            query=queries[0],  # Primary query
            results=all_results,
            total_found=len(all_results),
            filters_applied={"multi_query": True, "num_queries": len(queries)}
        )

    def format_context(
        self,
        results: list[SearchResult],
        max_chars: int = 4000,
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieval results as context for the LLM.

        Args:
            results: List of SearchResults
            max_chars: Maximum context length
            include_metadata: Include metadata headers

        Returns:
            Formatted context string
        """
        context_parts = []
        total_chars = 0

        for i, result in enumerate(results):
            # Build result text
            if include_metadata:
                header = f"[{result.metadata.get('function_name', 'Unknown')}]"
                if result.metadata.get('chunk_type'):
                    header += f" ({result.metadata['chunk_type']})"
                header += "\n"
            else:
                header = ""

            result_text = f"{header}{result.text}\n\n---\n\n"

            # Check if we have room
            if total_chars + len(result_text) > max_chars:
                # Try to fit a truncated version
                available = max_chars - total_chars - len(header) - 20
                if available > 100:
                    truncated = result.text[:available] + "..."
                    result_text = f"{header}{truncated}\n\n---\n\n"
                    context_parts.append(result_text)
                break

            context_parts.append(result_text)
            total_chars += len(result_text)

        return "".join(context_parts).strip()


class QueryPreprocessor:
    """
    Preprocessor for improving query quality.

    Can expand, rephrase, or decompose queries.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the preprocessor.

        Args:
            llm_client: Optional LLM client for advanced preprocessing
        """
        self.llm_client = llm_client

    def expand_query(self, query: str) -> list[str]:
        """
        Expand query into multiple variations.

        Simple version using keyword expansion.
        """
        variations = [query]

        # Add variations with common synonyms
        synonyms = {
            "fit": ["curve fit", "regression", "least squares"],
            "minimize": ["optimize", "find minimum", "optimization"],
            "integrate": ["integral", "quadrature", "integration"],
            "solve": ["find solution", "linear algebra"],
            "interpolate": ["interpolation", "estimate between points"],
            "filter": ["signal processing", "butterworth", "frequency"],
            "fft": ["fourier transform", "frequency analysis"],
            "sparse": ["sparse matrix", "compressed"],
            "distance": ["metric", "similarity", "euclidean"],
        }

        query_lower = query.lower()
        for keyword, expansions in synonyms.items():
            if keyword in query_lower:
                for expansion in expansions[:2]:  # Limit expansions
                    if expansion not in query_lower:
                        variations.append(f"{query} {expansion}")

        return variations[:5]  # Limit total variations

    def rephrase_for_code(self, query: str) -> str:
        """
        Rephrase query to be more code-focused.
        """
        # Add code-related terms if not present
        code_terms = ["scipy", "python", "function", "how to"]
        query_lower = query.lower()

        if not any(term in query_lower for term in code_terms):
            return f"scipy python {query}"

        return query


# Demonstration
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing Retriever...")

    # Create a simple vector store for testing
    store = VectorStore(collection_name="test_retriever")
    store.reset_collection()

    # Add test documents
    docs = [
        ("scipy.optimize.minimize(fun, x0) minimizes a scalar function using various methods like BFGS, Nelder-Mead",
         "minimize_summary", {"module": "scipy.optimize", "function_name": "minimize", "chunk_type": "summary"}),
        ("Example: from scipy.optimize import minimize; result = minimize(lambda x: x**2, [1.0])",
         "minimize_examples", {"module": "scipy.optimize", "function_name": "minimize", "chunk_type": "examples"}),
        ("scipy.optimize.curve_fit(f, xdata, ydata) fits a function to data using non-linear least squares",
         "curve_fit_summary", {"module": "scipy.optimize", "function_name": "curve_fit", "chunk_type": "summary"}),
        ("scipy.integrate.quad(func, a, b) computes a definite integral numerically",
         "quad_summary", {"module": "scipy.integrate", "function_name": "quad", "chunk_type": "summary"}),
    ]

    for text, id, meta in docs:
        store.add_document(text, id, meta)

    # Create retriever
    retriever = Retriever(store, default_top_k=3)

    # Test basic retrieval
    print("\n1. Basic retrieval:")
    result = retriever.retrieve("how to minimize a function")
    for r in result.results:
        print(f"  [{r.score:.3f}] {r.metadata['function_name']}: {r.text[:60]}...")

    # Test with context
    print("\n2. Retrieval with context:")
    result = retriever.retrieve_with_context("minimize")
    for r in result.results:
        print(f"  [{r.metadata['chunk_type']}] {r.metadata['function_name']}")

    # Test query expansion
    print("\n3. Query expansion:")
    preprocessor = QueryPreprocessor()
    variations = preprocessor.expand_query("fit curve to data")
    for v in variations:
        print(f"  - {v}")

    # Test context formatting
    print("\n4. Formatted context:")
    result = retriever.retrieve("optimization", top_k=2)
    context = retriever.format_context(result.results, max_chars=500)
    print(context[:300] + "...")
