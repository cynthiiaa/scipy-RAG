# Adding New SciPy Modules to the Knowledge Base

To add more modules (e.g., `signal`, `spatial`, `stats`), run these cells in **notebook 02** in order:

---

## Step 1: Setup (cell 2)

```python
import os
import sys
import json
from datetime import datetime, UTC
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / 'src'))

from dotenv import load_dotenv
load_dotenv()
```

---

## Step 2: Scrape new modules (cell 11)

Uncomment and update `modules_to_scrape`:

```python
scraper = SciPyDocsScraper(
    delay=0.5,
    output_dir=str(Path.cwd().parent / "data" / "raw")
)

modules_to_scrape = ["signal", "spatial", "stats"]  # <-- your new modules
live_docs = scraper.scrape_all(modules=modules_to_scrape)

print(f"\nScraped {len(live_docs)} documents from live site")
```

---

## Step 3: Load documents (cell 13)

```python
with open(Path.cwd().parent / "data" / "raw" / 'scipy_all.json', 'r') as f:
    documents = json.load(f)
```

---

## Step 4: Add provenance metadata (cell 14)

```python
now_iso = datetime.now(UTC).isoformat(timespec='seconds').replace("+00:00", "Z")

for scraped_doc in documents:
    scraped_doc.setdefault("retrieved_at", now_iso)
    scraped_doc.setdefault("source_url", None)
    scraped_doc.setdefault("scipy_doc_version", None)
```

---

## Step 5: Chunk documents (cell 29)

```python
all_chunks = chunk_documents(documents, strategy="code_aware", chunk_size=600, overlap=100)
print(f"Total chunks created: {len(all_chunks)}")
```

---

## Step 6: Propagate provenance to chunks (cell 30)

```python
by_title = {d.get("title"): d for d in documents}

for ch in all_chunks:
    fn = ch.metadata.get("function_name")
    doc = by_title.get(fn)
    if doc is None:
        continue
    ch.metadata.setdefault("source_url", doc.get("source_url"))
    ch.metadata.setdefault("retrieved_at", doc.get("retrieved_at"))
    ch.metadata.setdefault("scipy_doc_version", doc.get("scipy_doc_version"))
```

---

## Step 7: Embed and store (cell 36)

**Option A: Append to existing collection** (skip cell 33)

```python
# Get existing collection (don't delete it!)
collection = chroma_client.get_collection("scipy_docs")
```

**Option B: Full rebuild** (run cell 33 first to delete, then cell 36)

Then run the embedding loop:

```python
for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing chunks"):
    batch = all_chunks[i:i + batch_size]
    # ... (rest of cell 36)
```
