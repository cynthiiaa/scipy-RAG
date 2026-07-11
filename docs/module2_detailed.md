---
marp: true
size: 16:9
style: |
  section {
    font-size: 24px;
  }
  h1 {
    font-size: 42px;
  }
  h2 {
    font-size: 32px;
  }
  h3 {
    font-size: 28px;
  }
  pre {
    font-size: 18px;
  }
  table {
    font-size: 20px;
  }
math: mathjax
---

# Module 2: Building the SciPy Knowledge Base

**Prerequisites**: Completed Module 1, basic Python, understanding of embeddings

---

## 1. Module Goals

By the end of this module, you will:

1. **Scrape** SciPy documentation from the official website
2. **Clean** raw HTML into embedding-ready text
3. **Chunk** documents effectively for retrieval
4. **Build** a production-ready vector store
5. **Test** retrieval quality

---

## 2. Web Scraping Fundamentals

### Be a Good Citizen

When scraping any website, follow these principles:

| Principle         | Why It Matters               | Implementation                                  |
| ----------------- | ---------------------------- | ----------------------------------------------- |
| **robots.txt**    | Respects site owner's wishes | Check `https://docs.scipy.org/robots.txt` first |
| **Rate limiting** | Don't overwhelm servers      | Add at least 0.5s delay between requests        |
| **User-Agent**    | Identify yourself clearly    | `'SciPyRAGWorkshop/1.0 (Educational)'`          |
| **Caching**       | Avoid re-scraping            | Save data locally after first scrape            |

---

Before scraping or crawling a website for documents, check `robots.txt` to see whether the site allows automated crawling of those pages.

#### Example

```
User-agent: *
Disallow: /private/
Allow: /blog/
```

👆🏾 All crawlers: do not crawl /private/, but /blog/ is okay.

**`robots.txt` doesn't physically prevent access like how authentication or permissions would. It's more like a preference than a boundary.**

### Basic Scraper Setup

```python
class SciPyDocsScraper:
    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SciPyRAGWorkshop/1.0 (Educational)'
        })
```

---

## 3. What We Extract

> Review the scraper code (`src/scraper.py`).

From each SciPy function page, we extract:

| Field           | Example                                    |
| --------------- | ------------------------------------------ |
| **URL**         | `https://docs.scipy.org/.../minimize.html` |
| **Title**       | `scipy.optimize.minimize`                  |
| **Module**      | `scipy.optimize`                           |
| **Signature**   | `minimize(fun, x0, method=None, ...)`      |
| **Description** | Minimization of scalar function...         |
| **Parameters**  | `fun: callable, x0: ndarray, ...`          |
| **Returns**     | `OptimizeResult` object                    |
| **Examples**    | Code snippets                              |

---

### The Link Collector Function

The scraper has a **link collector** function that:

1. Starts on a SciPy module page (e.g., `integrate.html`)
2. Looks through every clickable link on that page
3. Finds individual documentation pages for functions and classes

**URL Processing Steps:**

```
Short link: "generated/scipy.integrate.quad.html"
    ↓ urljoin() → Full URL

Link with fragment: "scipy.integrate.quad.html#examples"
    ↓ urldefrag() → Remove fragment

Full URL: "https://docs.scipy.org/.../scipy.integrate.quad.html"
    ↓ urlparse() → Extract path for filtering
```

---

**Filtering Logic:**

- Only keeps links that are generated documentation pages
- Must belong to the requested module
- Must end in `.html`
- Must not already be collected (deduplication)

In plain English: The function opens the module's table-of-contents page, finds all useful function/class documentation links, cleans them up, removes duplicates, and returns the list so the scraper can visit each page next.

---

## 4. Data Freshness and Provenance

### What is Provenance?

**Provenance** = the source history of a document: where it came from, when you collected it, and what version it came from.

In a scraped knowledge base, provenance is basically the document's **receipt**.

### Example Provenance Fields

```python
{
    "source_url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html",
    "retrieved_at": "2024-01-15T10:30:00Z",
    "scipy_doc_version": "1.18.0"
}
```

---

### Why Provenance Matters

1. **Trust**: Every chunk can be traced back to its original source
2. **Refresh**: When SciPy updates their docs, you can compare your stored `retrieved_at` and `scipy_doc_version` to know what needs re-scraping
3. **Debugging**: When retrieval gives wrong results, you can verify the source

So when you see this comment in code:

```python
# Attach provenance to documents
```

It means: "Add metadata that tells us where each document came from and when we got it."

---

## 5. Data Cleaning: The Critical Step

### Garbage In, Garbage Out

Just like data science, RAG quality depends on clean data:

| Data Science                       | RAG                                 |
| ---------------------------------- | ----------------------------------- |
| Clean CSV/JSON before analysis     | Clean scraped text before embedding |
| Handle missing values              | Handle empty sections/broken HTML   |
| Normalize formats (dates, numbers) | Normalize whitespace, encoding      |
| Remove outliers                    | Remove boilerplate/nav text         |
| Feature engineering                | Chunking strategy                   |

---

### Common Scraping Issues

Raw scraped HTML often has:

- Excessive whitespace/newlines between tokens
- Broken code formatting (`>>>importnumpy` instead of `>>> import numpy`)
- HTML artifacts (`&nbsp;`, `&amp;`)
- Navigation/boilerplate text

---

### The SciPy-Specific Problem

SciPy's docs wrap almost every token in `<span>` or `<code>` tags for syntax highlighting. When BeautifulSoup extracts with `separator="\n"`, it puts a newline between each element:

```html
<code>integrate</code>.<code>tplquad</code>(<code>f</code>...
```

Becomes:

```
integrate
.
tplquad
(
f
...
```

---

**Fix**: Use space separator and normalize whitespace:

```python
# Before (broken)
full_text = main_content.get_text(separator="\n", strip=True)

# After (fixed)
full_text = main_content.get_text(separator=" ", strip=True)
full_text = re.sub(r'\s+', ' ', full_text)  # collapse multiple spaces/newlines
```

---

### Key Insight: Prose and Code Need Different Strategies

````python
def _clean_text_preserve_code(self, soup):
    # 1. Extract code blocks FIRST (preserve formatting)
    code_blocks = []
    for pre in soup.find_all('pre'):
        code_blocks.append(pre.get_text())  # Keep original spacing
        pre.replace_with(f"__CODE_BLOCK_{len(code_blocks)}__")

    # 2. Clean remaining text (normalize whitespace)
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text)

    # 3. Restore code blocks with proper formatting
    for i, code in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", f"\n```\n{code}\n```\n")

    return text
````

---

### Common Cleaning Tasks Reference

| Issue                              | Fix                                                 |
| ---------------------------------- | --------------------------------------------------- |
| Excessive whitespace/newlines      | `re.sub(r'\s+', ' ', text)`                         |
| HTML artifacts (`&nbsp;`, `&amp;`) | `html.unescape(text)`                               |
| Navigation/boilerplate text        | Remove headers, footers, sidebars before extraction |
| Broken code blocks                 | Preserve `<pre>` and `<code>` formatting separately |
| Unicode issues                     | `unicodedata.normalize('NFKC', text)`               |
| Duplicate content                  | Dedupe by hash or similarity                        |

---

### The Tradeoff: Scraping Speed vs Quality

|                    | Scraping (one-time) | Querying (many times)          |
| ------------------ | ------------------- | ------------------------------ |
| **Skip cleaning**  | Fast                | Poor embeddings, bad retrieval |
| **Clean properly** | Slower              | Better vectors, better results |

---

### Where to Clean: As Early as Possible

```
Scrape → Clean → Store raw → Chunk → Embed → Store vectors
             ↑
          Best place (clean once, use everywhere)
```

**Why clean before storing:**

1. **Clean once** — Don't repeat cleaning logic in every downstream step
2. **Better chunks** — Chunker works with clean text, makes better decisions about splits
3. **Better embeddings** — Clean text produces more meaningful vectors
4. **Smaller storage** — No wasted space on garbage characters
5. **Debuggable** — You can inspect the cleaned data before it enters the vector store

---

## 6. Document Chunking

### What is Chunking?

**Chunking** is the process of breaking documents into smaller sections so each section can be embedded and retrieved independently.

### Why Chunking Matters

Poor chunking leads to:

- Retrieved context missing key information
- Code examples split in the middle (useless!)
- Related information separated

---

### The Goal

Create chunks that are:

- **Complete** enough to be useful
- **Small** enough for embedding models (context limits)
- **Coherent** (don't split mid-thought or mid-code)

### Chunking Strategies Comparison

| Strategy       | Pros                  | Cons                   | Best For       |
| -------------- | --------------------- | ---------------------- | -------------- |
| **Fixed-size** | Simple, predictable   | May split mid-sentence | Uniform text   |
| **Recursive**  | Respects boundaries   | Uneven chunks          | Prose/articles |
| **Code-aware** | Preserves code blocks | More complex           | Documentation  |
| **Semantic**   | Coherent topics       | Expensive (uses LLM)   | Long documents |

Semantic chunking is meaning-based. It uses embeddings or model-based logic to detect topic shifts and group related sentences together.

---

### Fixed-Size Chunking

```python
def fixed_size_chunker(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Overlap for context continuity
    return chunks
```

**Problem**: May cut text mid-sentence or mid-code!

```
"scipy.optimize.minimize finds the mini|mum of a function"
                                       ^ Awkward split!
```

---

### Recursive Text Splitting

Tries separators in order of preference:

```python
separators = ["\n\n", "\n", ". ", ", ", " ", ""]

# First try to split on paragraphs
# Then sentences
# Then words
# Finally characters (last resort)
```

Recursive chunking is rule-based. It keeps splitting text until each chunk fits within a target size, while trying to preserve natural boundaries like paragraphs and sentences.

---

### Code-Aware Chunking

Keep code blocks intact!

````python
def code_aware_chunker(text: str, chunk_size: int = 800):
    # Detect code blocks (indented or fenced)
    code_pattern = r'(```[\s\S]*?```|(?:^(?:    |\t).*$\n?)+)'

    # Split into code and non-code segments
    parts = re.split(code_pattern, text)

    # Process each part, keeping code together
    for part in parts:
        if is_code_block(part):
            # Try to keep code intact
            if len(current_chunk) + len(part) <= chunk_size:
                current_chunk += part
            else:
                # Code too large, must split (not ideal)
                ...
````

---

## 7. Our Chunking Strategy for SciPy

Create **multiple chunk types** per document:

1. **Summary Chunk**: Signature + description + key parameters
2. **Examples Chunk**: All code examples (kept together)
3. **Full-text Chunks**: Remaining content with overlap

```python
chunks = [
    Chunk(text=summary, chunk_type="summary", ...),
    Chunk(text=examples, chunk_type="examples", ...),
    Chunk(text=full_part_1, chunk_type="full_text", ...),
    ...
]
```

This allows **targeted retrieval** by chunk type.

---

### Example: curve_fit Results

When you search for `curve_fit`, you might get the same function 3 times with different chunk types:

- `scipy_optimize_curve_fit::examples`
- `scipy_optimize_curve_fit::full_text`
- `scipy_optimize_curve_fit::summary`

This is expected since each chunk type serves a different purpose.

---

## 8. Metadata Schema

Store rich metadata for filtering:

```python
metadata = {
    "module": "scipy.optimize",
    "function_name": "minimize",
    "doc_type": "function",      # or "class"
    "chunk_type": "summary",     # or "examples", "full_text"
    "url": "https://docs.scipy.org/..."
}
```

### Benefits of Rich Metadata

- Filter by module: `where={"module": "scipy.optimize"}`
- Filter by chunk type: `where={"chunk_type": "examples"}`
- Track source for citations

---

## 9. Building the Vector Store

> Review the vector store code (`src/vectorstore.py`)?

```python
from tqdm import tqdm

# Chunk all documents
all_chunks = chunk_documents(documents, strategy="code_aware")

# Add to ChromaDB in batches
batch_size = 50
for i in tqdm(range(0, len(all_chunks), batch_size)):
    batch = all_chunks[i:i + batch_size]

    collection.add(
        ids=[chunk.chunk_id for chunk in batch],
        documents=[chunk.text for chunk in batch],
        metadatas=[chunk.metadata for chunk in batch]
    )

print(f"Added {collection.count()} chunks to vector store")
```

---

### What This Code Does

1. Splits each SciPy document into smaller overlapping chunks
2. Gives each chunk a unique ID
3. Keeps metadata linking it back to the original document
4. Stores those chunks in a Chroma collection for retrieval

The goal is to improve RAG search by retrieving the most relevant **section** of a document instead of the entire document.

---

## 10. Testing Retrieval Quality

```python
def test_retrieval(query: str, expected_function: str):
    results = collection.query(query_texts=[query], n_results=5)

    retrieved_functions = [
        r['function_name'] for r in results['metadatas'][0]
    ]

    if expected_function in retrieved_functions:
        rank = retrieved_functions.index(expected_function) + 1
        print(f"✓ Found '{expected_function}' at rank {rank}")
    else:
        print(f"✗ Did not find '{expected_function}'")
        print(f"  Retrieved: {retrieved_functions}")
```

---

## 11. Understanding Out-of-Corpus Queries

### What is an Out-of-Corpus Query?

An **out-of-corpus query** is when the user's desired answer is not present in the indexed knowledge base, even if the answer exists elsewhere.

### Example: Linear Equations Query

**Query**: "solve a system of linear equations"

**Expected**: `scipy.linalg.solve` or `scipy.sparse.linalg`

**Problem**: Your corpus only contains `scipy.optimize` and `scipy.integrate`

**What happens**: The retriever can only choose the nearest available neighbors, which happen to be nonlinear equation solvers like `root`, `BroydenFirst`, and `KrylovJacobian`.

The system is not "bad" at retrieval. It's being forced to answer from an incomplete knowledge base.

---

### Example: Butterworth Filter Query

**Query**: "filter a signal with butterworth"

**Results**:

```
nquad                         scipy.integrate       Distance: ~0.70
elementwise.find_minimum      scipy.optimize        Distance: ~0.70
direct                        scipy.optimize        Distance: ~0.70
```

The retriever has no access to `scipy.signal`, where Butterworth filtering lives.

**Diagnostic insight**: The distances (~0.70) are worse than other queries (~0.45-0.49), indicating the corpus at least had adjacent material for those queries. Here, the top results are not even meaningfully related.

---

### The Key Lesson

RAG quality depends as much on **corpus coverage** as on embeddings, chunking, or reranking.

A strong system should detect when the retrieved evidence is merely adjacent rather than actually sufficient. Instead of confidently answering with the best available but wrong docs, it should surface a fallback like:

> "The indexed corpus does not appear to include the relevant SciPy linear algebra docs."

This is where **retrieval confidence**, **module coverage checks**, **lexical intent checks**, and **abstention policies** become important.

**In production RAG, knowing when NOT to answer is just as valuable as retrieving the right chunk.**

---

## 12. Debugging Poor Retrieval

| Problem          | Symptom             | Solution                                    |
| ---------------- | ------------------- | ------------------------------------------- |
| Chunks too small | Missing context     | Increase `chunk_size`                       |
| Chunks too large | Diluted content     | Decrease `chunk_size`                       |
| Code split       | Broken examples     | Use code-aware chunking                     |
| Wrong results    | Irrelevant docs     | Check embedding model, add metadata filters |
| High distances   | Corpus coverage gap | Add more modules to knowledge base          |

---

## 13. Module 2 Summary

### What We Built

1. **Scraper**: Fetches SciPy docs (or uses sample data)
2. **Cleaner**: Preserves code blocks, normalizes whitespace
3. **Chunker**: Code-aware strategy with multiple chunk types
4. **Vector Store**: Persistent ChromaDB with rich metadata

---

### Key Takeaways

- Be respectful when scraping (rate limits, caching, User-Agent)
- Clean data early —> garbage in, garbage out
- Chunking strategy significantly impacts retrieval quality
- Metadata enables powerful filtering
- Always test retrieval before moving on
- Understand corpus coverage limitations

---

## Quick Reference

### Cleaning Function Template

```python
def clean_scraped_text(text: str) -> str:
    import re
    import html

    text = html.unescape(text)           # Fix HTML entities
    text = re.sub(r'\s+', ' ', text)     # Collapse whitespace
    text = re.sub(r'\[\d+\]', '', text)  # Remove citation markers
    text = text.strip()
    return text
```

---

### Chunking Template

```python
def chunk_text(text: str, max_chars: int = 700, overlap: int = 120) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks
```

---

### Metadata Filter Examples

```python
# Filter by module
results = collection.query(
    query_texts=["optimization"],
    where={"module": "scipy.optimize"}
)

# Filter by chunk type
results = collection.query(
    query_texts=["example code"],
    where={"chunk_type": "examples"}
)

# Combined filter
results = collection.query(
    query_texts=["minimize function"],
    where={
        "$and": [
            {"module": "scipy.optimize"},
            {"chunk_type": "summary"}
        ]
    }
)
```
