"""
Document Chunking Utilities

This module provides various chunking strategies for preparing
documents for embedding and storage in a vector database.
"""

import re
from dataclasses import dataclass
from typing import Callable

import tiktoken


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    text: str
    chunk_id: str
    source_id: str
    chunk_index: int
    metadata: dict

    def __len__(self):
        return len(self.text)


class TokenCounter:
    """
    Utility for counting tokens using tiktoken.

    Useful for ensuring chunks don't exceed model context limits.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize the token counter.

        Args:
            model: Model name for tokenization
        """
        # Use cl100k_base encoding (used by text-embedding-3-small, GPT-4)
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])


def fixed_size_chunker(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    token_counter: TokenCounter = None
) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.

    This is the simplest chunking strategy. Good for uniform text
    but may split code or sentences awkwardly.

    Args:
        text: Text to chunk
        chunk_size: Target characters per chunk (or tokens if token_counter provided)
        overlap: Characters/tokens of overlap between chunks
        token_counter: Optional TokenCounter for token-based chunking

    Returns:
        List of text chunks
    """
    # Basic parameter validation to avoid infinite loops
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    if token_counter:
        # Token-based chunking
        tokens = token_counter.encoding.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = token_counter.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap

        return chunks
    else:
        # Character-based chunking
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

        return chunks


def recursive_text_splitter(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: list[str] = None
) -> list[str]:
    """
    Recursively split text using a hierarchy of separators.

    Tries to split on larger units (paragraphs) first, then falls back
    to smaller units (sentences, words) if chunks are too large.

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Characters of overlap
        separators: List of separators to try, in order of preference

    Returns:
        List of text chunks
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", ", ", " ", ""]

    def _split(text: str, sep_idx: int = 0) -> list[str]:
        if sep_idx >= len(separators):
            # No more separators, force split
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        separator = separators[sep_idx]

        if separator == "":
            # Split by character as last resort
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        parts = text.split(separator)

        chunks = []
        current_chunk = ""

        for part in parts:
            # Add separator back (except for the first part)
            test_part = part if not current_chunk else separator + part

            if len(current_chunk) + len(test_part) <= chunk_size:
                current_chunk += test_part
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # If part itself is too large, recursively split
                if len(part) > chunk_size:
                    sub_chunks = _split(part, sep_idx + 1)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    return _split(text)


def code_aware_chunker(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100
) -> list[str]:
    """
    Chunk text while trying to keep code blocks intact.

    This is crucial for documentation that contains code examples.
    Splitting a code block in the middle makes it useless.

    Args:
        text: Text to chunk (may contain code blocks)
        chunk_size: Maximum characters per chunk
        overlap: Characters of overlap for non-code parts

    Returns:
        List of text chunks
    """
    # Pattern for code blocks (indented or fenced)
    code_block_pattern = r'(```[\s\S]*?```|(?:^(?:    |\t).*$\n?)+)'

    # Split into code and non-code segments
    parts = re.split(code_block_pattern, text, flags=re.MULTILINE)

    chunks = []
    current_chunk = ""

    for part in parts:
        is_code = part.startswith('```') or part.startswith('    ') or part.startswith('\t')

        if is_code:
            # Try to keep code blocks intact
            if len(current_chunk) + len(part) <= chunk_size:
                current_chunk += part
            else:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # If code block fits in a chunk, add it
                if len(part) <= chunk_size:
                    current_chunk = part
                else:
                    # Code block too large, split it (not ideal but necessary)
                    code_chunks = fixed_size_chunker(part, chunk_size, overlap)
                    chunks.extend(code_chunks[:-1])
                    current_chunk = code_chunks[-1] if code_chunks else ""
        else:
            # For non-code text, use recursive splitting
            if len(current_chunk) + len(part) <= chunk_size:
                current_chunk += part
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Split the non-code part
                text_chunks = recursive_text_splitter(part, chunk_size, overlap)
                if text_chunks:
                    chunks.extend(text_chunks[:-1])
                    current_chunk = text_chunks[-1]
                else:
                    current_chunk = ""

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def semantic_chunker(
    text: str,
    chunk_size: int = 500,
    similarity_threshold: float = 0.7,
    embedding_fn: Callable[[str], list[float]] = None
) -> list[str]:
    """
    Chunk text based on semantic similarity between sentences.

    Groups semantically similar sentences together. Requires an
    embedding function. More expensive but creates more coherent chunks.

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        similarity_threshold: Similarity threshold for grouping (0-1)
        embedding_fn: Function that takes text and returns embedding vector

    Returns:
        List of text chunks
    """
    import numpy as np

    if embedding_fn is None:
        # Fall back to recursive splitter
        return recursive_text_splitter(text, chunk_size)

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return sentences

    # Get embeddings for all sentences
    embeddings = [embedding_fn(s) for s in sentences]
    embeddings = np.array(embeddings)

    # Normalize for cosine similarity (defensive against zero norms)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    eps = 1e-12
    norms = np.maximum(norms, eps)
    embeddings = embeddings / norms

    # Group sentences by semantic similarity
    chunks: list[str] = []
    current_chunk = [sentences[0]]
    current_indices = [0]
    current_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        similarity = float(np.dot(current_embedding, embeddings[i]))
        candidate_text = " ".join(current_chunk + [sentences[i]])

        if similarity >= similarity_threshold and len(candidate_text) <= chunk_size:
            current_chunk.append(sentences[i])
            current_indices.append(i)
            current_embedding = np.mean(embeddings[current_indices], axis=0)
            norm = float(np.linalg.norm(current_embedding))
            if norm > 0:
                current_embedding = current_embedding / norm
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_indices = [i]
            current_embedding = embeddings[i]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_scipy_document(
    doc: dict,
    strategy: str = "code_aware",
    chunk_size: int = 800,
    overlap: int = 100
) -> list[Chunk]:
    """
    Chunk a SciPy documentation document.

    Creates structured chunks optimized for RAG retrieval.

    Args:
        doc: Document dictionary with 'full_text', 'signature', 'examples', etc.
        strategy: Chunking strategy ('fixed', 'recursive', 'code_aware')
        chunk_size: Maximum characters per chunk
        overlap: Characters of overlap

    Returns:
        List of Chunk objects
    """
    chunks = []
    source_id = doc.get('function_name', doc.get('title', 'unknown'))

    # Strategy 1: Create a "summary" chunk with signature + description
    summary = f"{doc.get('signature', '')}\n\n{doc.get('description', '')}"
    if doc.get('parameters'):
        summary += f"\n\nParameters:\n{doc['parameters'][:500]}"

    chunks.append(Chunk(
        text=summary.strip(),
        chunk_id=f"{source_id}_summary",
        source_id=source_id,
        chunk_index=0,
        metadata={
            "module": doc.get('module', ''),
            "function_name": doc.get('function_name', ''),
            "doc_type": doc.get('doc_type', 'function'),
            "chunk_type": "summary",
            "url": doc.get('url', ''),
            "source_url": doc.get("source_url", doc.get("url", "")),
            "retrieved_at": doc.get("retrieved_at", ""),
            "scipy_doc_version": doc.get("scipy_doc_version", "")
        }
    ))

    # Strategy 2: Create separate chunk for examples (if they exist and are substantial)
    examples = doc.get('examples', '')
    if examples and len(examples) > 50:
        chunks.append(Chunk(
            text=f"Examples for {source_id}:\n\n{examples}",
            chunk_id=f"{source_id}_examples",
            source_id=source_id,
            chunk_index=1,
            metadata={
                "module": doc.get('module', ''),
                "function_name": doc.get('function_name', ''),
                "doc_type": doc.get('doc_type', 'function'),
                "chunk_type": "examples",
                "url": doc.get('url', '')
            }
        ))

    # Strategy 3: Chunk the full text for comprehensive coverage
    full_text = doc.get('full_text', '')
    if full_text and len(full_text) > chunk_size:
        if strategy == "fixed":
            text_chunks = fixed_size_chunker(full_text, chunk_size, overlap)
        elif strategy == "recursive":
            text_chunks = recursive_text_splitter(full_text, chunk_size, overlap)
        else:  # code_aware
            text_chunks = code_aware_chunker(full_text, chunk_size, overlap)

        for i, text in enumerate(text_chunks):
            chunks.append(Chunk(
                text=text,
                chunk_id=f"{source_id}_full_{i}",
                source_id=source_id,
                chunk_index=len(chunks),
                metadata={
                    "module": doc.get('module', ''),
                    "function_name": doc.get('function_name', ''),
                    "doc_type": doc.get('doc_type', 'function'),
                    "chunk_type": "full_text",
                    "url": doc.get('url', ''),
            "source_url": doc.get("source_url", doc.get("url", "")),
            "retrieved_at": doc.get("retrieved_at", ""),
            "scipy_doc_version": doc.get("scipy_doc_version", "")
        }
            ))

    return chunks


def chunk_documents(
    documents: list[dict],
    strategy: str = "code_aware",
    chunk_size: int = 800,
    overlap: int = 100
) -> list[Chunk]:
    """
    Chunk multiple documents.

    Args:
        documents: List of document dictionaries
        strategy: Chunking strategy
        chunk_size: Maximum characters per chunk
        overlap: Characters of overlap

    Returns:
        List of all Chunks
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_scipy_document(doc, strategy, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks


# Demonstration
if __name__ == "__main__":
    # Example text with code
    sample_text = """scipy.optimize.minimize finds the minimum of a function.

This function supports multiple optimization methods including:
- Nelder-Mead
- BFGS
- L-BFGS-B

Example usage:

    from scipy.optimize import minimize

    def objective(x):
        return x[0]**2 + x[1]**2

    result = minimize(objective, [1, 1])
    print(result.x)

The result object contains the optimal parameters and convergence information.

You can also specify bounds on the variables:

    from scipy.optimize import minimize

    bounds = [(0, 10), (0, 10)]
    result = minimize(objective, [5, 5], bounds=bounds)
"""

    print("Fixed size chunks:")
    print("-" * 40)
    for i, chunk in enumerate(fixed_size_chunker(sample_text, 200, 20)):
        print(f"[{i}] {chunk[:50]}... ({len(chunk)} chars)")

    print("\nRecursive chunks:")
    print("-" * 40)
    for i, chunk in enumerate(recursive_text_splitter(sample_text, 200, 20)):
        print(f"[{i}] {chunk[:50]}... ({len(chunk)} chars)")

    print("\nCode-aware chunks:")
    print("-" * 40)
    for i, chunk in enumerate(code_aware_chunker(sample_text, 300, 50)):
        print(f"[{i}] {chunk[:50]}... ({len(chunk)} chars)")
