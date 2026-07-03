"""
Add new SciPy modules to the knowledge base.

Usage:
    python add_modules.py signal spatial stats
    python add_modules.py --list                 # Show available modules
    python add_modules.py --rebuild optimize     # Delete and rebuild collection

Examples:
    # Add signal processing and stats modules
    python add_modules.py signal stats

    # Rebuild entire collection with just optimize and integrate
    python add_modules.py --rebuild optimize integrate
"""

import argparse
import sys
from datetime import datetime, UTC
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from scraper import SciPyDocsScraper
from chunker import chunk_documents


def get_chroma_client():
    """Initialize ChromaDB client."""
    import chromadb
    chroma_path = Path(__file__).parent.parent / "chroma_db"
    chroma_path.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))


def get_openai_client():
    """Initialize OpenAI client for embeddings."""
    from openai import OpenAI
    return OpenAI()


def embed_texts(client, texts, model="text-embedding-3-small"):
    """Embed a list of texts using OpenAI."""
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def clean_metadata(metadata):
    """Replace None values with empty strings for ChromaDB."""
    return {key: "" if value is None else value for key, value in metadata.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Add SciPy modules to the knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "modules",
        nargs="*",
        help="Modules to add (e.g., signal spatial stats)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available modules"
    )
    parser.add_argument(
        "--rebuild", "-r",
        action="store_true",
        help="Delete existing collection and rebuild from scratch"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="Chunk size in characters (default: 600)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)"
    )

    args = parser.parse_args()

    # List available modules
    if args.list:
        print("Available SciPy modules:")
        for module in SciPyDocsScraper.MODULES:
            print(f"  - {module}")
        return

    if not args.modules:
        parser.print_help()
        sys.exit(1)

    # Validate modules
    valid_modules = set(SciPyDocsScraper.MODULES)
    for mod in args.modules:
        if mod not in valid_modules:
            print(f"Error: Unknown module '{mod}'")
            print(f"Run 'python add_modules.py --list' to see available modules")
            sys.exit(1)

    print(f"Adding modules: {', '.join(args.modules)}")
    print("=" * 50)

    # Step 1: Scrape
    print("\n[1/4] Scraping documentation...")
    scraper = SciPyDocsScraper(
        delay=args.delay,
        output_dir=str(Path(__file__).parent.parent / "data" / "raw")
    )
    documents = scraper.scrape_all(modules=args.modules)
    print(f"      Scraped {len(documents)} documents")

    # Convert ScrapedDocument objects to dicts if needed
    if documents and hasattr(documents[0], '__dict__'):
        documents = [doc.__dict__ if hasattr(doc, '__dict__') else doc for doc in documents]

    # Step 2: Add provenance
    print("\n[2/4] Adding provenance metadata...")
    now_iso = datetime.now(UTC).isoformat(timespec='seconds').replace("+00:00", "Z")
    for doc in documents:
        doc.setdefault("retrieved_at", now_iso)
        doc.setdefault("source_url", None)
        doc.setdefault("scipy_doc_version", None)

    # Step 3: Chunk
    print("\n[3/4] Chunking documents...")
    all_chunks = chunk_documents(documents, strategy="code_aware", chunk_size=args.chunk_size, overlap=100)
    print(f"      Created {len(all_chunks)} chunks")

    # Propagate provenance to chunks
    by_title = {d.get("title"): d for d in documents}
    for ch in all_chunks:
        fn = ch.metadata.get("function_name")
        doc = by_title.get(fn)
        if doc:
            ch.metadata.setdefault("source_url", doc.get("source_url"))
            ch.metadata.setdefault("retrieved_at", doc.get("retrieved_at"))
            ch.metadata.setdefault("scipy_doc_version", doc.get("scipy_doc_version"))

    # Step 4: Embed and store
    print("\n[4/4] Embedding and storing in ChromaDB...")
    chroma_client = get_chroma_client()
    openai_client = get_openai_client()

    if args.rebuild:
        print("      Deleting existing collection...")
        try:
            chroma_client.delete_collection("scipy_docs")
        except Exception:
            pass
        collection = chroma_client.create_collection(
            name="scipy_docs",
            metadata={"hnsw:space": "cosine"}
        )
    else:
        try:
            collection = chroma_client.get_collection("scipy_docs")
            print(f"      Appending to existing collection ({collection.count()} docs)")
        except Exception:
            collection = chroma_client.create_collection(
                name="scipy_docs",
                metadata={"hnsw:space": "cosine"}
            )
            print("      Created new collection")

    # Add chunks in batches
    batch_size = 50
    from tqdm import tqdm

    for i in tqdm(range(0, len(all_chunks), batch_size), desc="      Indexing"):
        batch = all_chunks[i:i + batch_size]

        ids = [chunk.chunk_id for chunk in batch]
        docs = [chunk.text for chunk in batch]
        metas = [clean_metadata(chunk.metadata) for chunk in batch]
        embs = embed_texts(openai_client, docs)

        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    print(f"\n{'=' * 50}")
    print(f"Done! Collection now has {collection.count()} documents")


if __name__ == "__main__":
    main()
