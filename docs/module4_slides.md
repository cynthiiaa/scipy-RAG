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

# Module 4: Evaluation & Capstone App

**Prerequisites:** ChromaDB populated with SciPy docs

---

## Module Goals

1. **Evaluate** RAG system quality with metrics
2. **Build** a Gradio web application
3. **Understand** production considerations

---

## 1. Why Evaluate?

"If you can't measure it, you can't improve it."

RAG systems can fail in multiple ways:

- **Retrieval failures**: Wrong documents retrieved
- **Generation failures**: Hallucinations, off-topic answers
- **Integration failures**: Good docs + good LLM = bad answer

We need metrics for both retrieval **AND** generation.

Retrieval metrics evaluate whether the right context was found while generation metrics evaluate the LLM's response.

---

## 2. Retrieval Metrics

Retrieval metrics tell us whether the RAG system found the right information and whether it ranked that information high enough to be useful.

### Precision@K

Of the top k results we retrieved, how many were actually relevant?

A high Precision@k means most of the top results are useful; a low Precision@k means the retriever is returning a lot of noise.

**⚠️ Irrelevant chunks can distract the LLM and reduce answer quality**

---

$$ \text{Precision@}k = \frac{\text{Number of relevant results in the top }k}{k} $$

```
Precision@5 = (Relevant docs in top 5) / 5
```

If you retrieve 5 documents and 3 are relevant, Precision@5 = 0.6.

---

### Recall@K

Of all the relevant results that exist, how many did we retrieve in the top k? This measures how much of the relevant information the retriever managed to find.

$$ \text{Recall@}k = \frac{\text{Number of relevant results in the top }k} {\text{Total number of relevant results}} $$

```
Recall@5 = (Relevant docs in top 5) / (Total relevant docs)
```

If there are 4 relevant documents total and top 5 retrieval contains 3 of them, Recall@5 = 0.75.

---

### Mean Reciprocal Rank (MRR)

MRR tells us how quickly the retriever finds the first useful result. A score closer to 1 means relevant information usually appears near the top.

For one query:
$$ \text{Reciprocal Rank} = \frac{1}{\text{rank of the first relevant result}} $$

‼️ Reciprocal Rank measures how high the first relevant result appears for a single query.

MRR averages those reciprocal ranks:
$$ \text{MRR} = \frac{1}{N} \sum\_{i=1}^{N} \frac{1}{\operatorname{rank}\_i} $$

**Example**
If the first relevant result is at position 3, MRR = 1/3 = 0.33

---

### 2.4 Hit@K

Did at least one relevant result appear in the top k results? Hit@k only checks whether a relevant result was found. It does not care how many relevant results were found or exactly where within the top k they appeared.

$$ \operatorname{Hit@}k = \begin{cases} 1, & \text{if at least one relevant result appears in the top } k \\ 0, & \text{otherwise.} \end{cases} $$

**Example**

```
Top 3 results:
1. Not relevant
2. Relevant
3. Not relevant
```

Since one relevant result appears in the top 3:
$$ \text{ Hit@3 } = 1 $$

If the first relevant result were ranked fourth:
$$ \text{ Hit@3 } = 0 $$

---

$$ \operatorname{HitRate@}k = \frac{\text{Number of queries with a hit in the top k​}}{\text{total number of queries}} $$

---

## 3. Retrieval Evaluation Implementation

```python
def evaluate_retrieval(question, expected_functions, retriever):
    results = retriever.retrieve(question, top_k=5)

    retrieved = [r.metadata['function_name'] for r in results]
    expected = set(expected_functions)

    # Precision
    relevant_retrieved = sum(1 for f in retrieved if f in expected)
    precision = relevant_retrieved / len(retrieved)

    # Recall
    recall = relevant_retrieved / len(expected)

    # MRR
    mrr = 0
    for i, f in enumerate(retrieved):
        if f in expected:
            mrr = 1 / (i + 1)
            break

    return {"precision": precision, "recall": recall, "mrr": mrr}
```

---

### Understanding the Evaluation Set

Each test item has:

```python
{
    "q": "How do I fit a curve to data?",
    "expected_parent": "scipy_optimize_curve_fit"
}
```

When someone asks "How do I fit a curve to data?", we expect the retriever to return a chunk from the `scipy_optimize_curve_fit` document.

**The evaluation checks retrieval quality only**. It does not evaluate the final LLM answer. It only checks whether the retriever can find the expected document/chunk within the top K results.

---

## 4. Generation Metrics

| Metric           | Question                          |
| ---------------- | --------------------------------- |
| **Faithfulness** | Is answer supported by context?   |
| **Relevance**    | Does answer address the question? |
| **Completeness** | Is answer thorough?               |
| **Code Quality** | Is the code correct and runnable? |

### Evaluation Approaches

1. **Human evaluation**: Gold standard but expensive
2. **LLM-as-judge**: Use GPT-4 to evaluate responses
3. **Automated metrics**: [RAGAS framework](https://docs.ragas.io/en/stable/)

---

## 5. LLM-as-Judge

An LLM judge can evaluate qualities that require understanding meaning. Also, a judge model can review hundreds or thousands of generated answers much faster and more cheaply than human reviewers.

```python
def evaluate_generation(question, answer, context):
    eval_prompt = f"""Evaluate this RAG response:

Question: {question}
Context: {context}
Answer: {answer}

Score 1-5 on:
1. Faithfulness: Is answer supported by context?
2. Relevance: Does answer address the question?
3. Completeness: Is answer thorough?
4. Code Quality: Is code correct? (N/A if no code)

Return JSON: {{"faithfulness": N, "relevance": N, ...}}"""

    return llm.generate(eval_prompt)
```

---

### Temperature Setting for Evaluation

When using LLM-as-judge, use `temperature=0` for deterministic, consistent outputs. This ensures reproducible evaluation scores.

- `temperature=0`: Deterministic, consistent outputs
- `temperature=0.3`: Slight variation, acceptable for code
- `temperature=0.7+`: More creative, not recommended for evaluation

---

## 6. Building Evaluation Datasets

```json
{
  "questions": [
    {
      "id": "q1",
      "question": "How do I minimize a function in SciPy?",
      "expected_functions": ["minimize"],
      "expected_modules": ["scipy.optimize"],
      "difficulty": "easy"
    },
    {
      "id": "q2",
      "question": "I need to fit an exponential curve to data",
      "expected_functions": ["curve_fit"],
      "expected_modules": ["scipy.optimize"],
      "difficulty": "easy"
    }
  ]
}
```

---

A good evaluation set should include:

- Questions at different difficulty levels
- Coverage across different modules
- Both exact function lookups and conceptual questions

---

## 7. Gradio Basics

Gradio makes ML demos easy:

```python
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Name"),
    outputs=gr.Textbox(label="Greeting")
)

demo.launch()
```

Gradio is basically a wrapper around your Python functions. Users often prefer working with a UI over the CLI or API endpoints.

---

## 8. Gradio for RAG

### Simple Interface

```python
def query_scipy(question: str) -> str:
    response = rag.query(question)
    return response.answer

demo = gr.Interface(
    fn=query_scipy,
    inputs=gr.Textbox(label="Question", placeholder="How do I..."),
    outputs=gr.Markdown(label="Answer"),
    title="SciPy RAG Assistant",
    examples=[
        ["How do I minimize a function?"],
        ["What's the best way to fit a curve?"]
    ]
)

demo.launch()
```

---

### Advanced Interface with Blocks

```python
with gr.Blocks() as demo:
    gr.Markdown("# SciPy RAG Assistant")

    with gr.Row():
        question = gr.Textbox(label="Question")
        model = gr.Dropdown(["GPT-4o-mini", "Ollama"], label="Model")

    submit = gr.Button("Ask")

    with gr.Row():
        answer = gr.Markdown(label="Answer")
        sources = gr.Markdown(label="Sources")

    submit.click(
        fn=query_with_sources,
        inputs=[question, model],
        outputs=[answer, sources]
    )
```

---

### The `query_with_sources` Function

This function:

1. Switches LLM if user picked a different model (GPT-4o-mini vs Ollama)
2. Queries the RAG system with the user's question
3. Formats the output into three parts:
   - The answer (markdown)
   - Sources list (which functions/docs were used)
   - Retrieved context (the raw chunks, if checkbox is ticked)

| `simple_query`          | `query_with_sources`               |
| ----------------------- | ---------------------------------- |
| Returns just the answer | Returns answer + sources + context |
| Uses default model      | Lets user switch models            |
| Minimal UI              | More controls                      |

---

## 9. Production Considerations

### Caching

Avoid re-computing embeddings for the same text:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> tuple:
    return tuple(embedding_model.embed(text))
```

A simple cache uses an MD5 hash of the text as the key. When a user asks a similar question twice, instead of calling the OpenAI embedding API again (which costs money and time), it looks up the cached result using an MD5 hash of the text as the key. The cache has a max size (1000 entries) and uses FIFO eviction. When full, it removes the oldest entry to make room.

**Note**: This is often a demo/illustration pattern. In production, you'd integrate this into your EmbeddingProvider class or use a persistent cache like Redis.

---

### Rate Limiting

Prevents overwhelming external APIs with too many requests:

```python
import time

class RateLimiter:
    def __init__(self, calls_per_minute: int = 60):
        self.delay = 60 / calls_per_minute
        self.last_call = 0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()
```

OpenAI and other providers enforce rate limits. If you exceed the limit, requests fail with a 4XX error. A rate limiter tracks when you last made a call and waits if necessary before allowing the next one. This keeps your app running smoothly during bursts of user queries.

---

**Example usage:**

```python
limiter = RateLimiter(calls_per_minute=60)

for question in user_questions:
    limiter.wait()  # Pauses if needed
    response = openai_client.embeddings.create(...)
```

---

### Keeping Docs Updated

Strategies:

1. **Scheduled re-scraping** (weekly/monthly cron job)
2. **Version tracking** (store SciPy version with docs)
3. **Incremental updates** (only re-embed changed docs)
4. **Hybrid approach** (static base + real-time web search)

```python
def needs_update(doc_url: str, stored_hash: str) -> bool:
    current_hash = hash_page(fetch(doc_url))
    return current_hash != stored_hash
```

---

## 10. ChromaDB Configuration

### HNSW Space Parameter

```python
collection = client.create_collection(
    name="scipy_docs",
    metadata={"hnsw:space": "cosine"}
)
```

This tells ChromaDB how to measure similarity between embeddings:

- `cosine`: Measures angle between vectors (most common for text embeddings)
- `l2`: Euclidean distance (straight-line distance)
- `ip`: Inner product (dot product)

OpenAI embeddings are normalized, so cosine is the standard choice. Two texts about the same topic will have vectors pointing in similar directions -> small angle -> high similarity.

---

### What is HNSW?

**HNSW** = Hierarchical Navigable Small World - the algorithm ChromaDB uses for fast similarity search.

Instead of comparing your query against every single vector (🐌), HNSW builds a graph structure that lets it jump to the right neighborhood quickly.

`hnsw:space` = which distance metric HNSW should use when building that graph and searching.

So `{"hnsw:space": "cosine"}` means: "build the search index using cosine similarity."

---

## 11. Switching Embedding Models

If you want to switch to a local embedding model like `nomic-embed-text` **you must re-index**!

| Model                  | Dimensions | Provider |
| ---------------------- | ---------- | -------- |
| text-embedding-3-small | 1536       | OpenAI   |
| nomic-embed-text       | 768        | Ollama   |

---

### Steps to Switch

1. Pull the model:

   ```bash
   ollama pull nomic-embed-text
   ```

2. Delete the existing ChromaDB:

   ```bash
   rm -rf /path/to/chroma_db
   ```

---

3. Re-run notebook 02 using OllamaEmbeddings:

   ```python
   from embeddings import OllamaEmbeddings

   embedding_provider = OllamaEmbeddings(model="nomic-embed-text")

   vector_store = VectorStore(
       collection_name="scipy_docs",
       persist_directory=str(CHROMA_PATH),
       embedding_provider=embedding_provider
   )
   ```

4. Update rag.py to use Ollama embeddings by default (or make it configurable).

---

## 12. Mixing Embedding and LLM Providers

You can use different providers for embeddings and LLM generation because they are completely separate steps:

```
┌─────────────────────────────────────────────────────────┐
│  RETRIEVAL (embeddings)                                 │
│                                                         │
│  Query ──► OpenAI embed ──► vector ──► ChromaDB search  │
│                                              │          │
│                                    retrieved text chunks│
└──────────────────────────────────────────────┼──────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────┐
│  GENERATION (LLM)                                       │
│                                                         │
│  text chunks + question ──► Claude/Ollama/GPT ──► answer│
└─────────────────────────────────────────────────────────┘
```

---

The LLM never sees the embedding vectors - it only sees plain text (the retrieved chunks). So you can mix:

- OpenAI embeddings + Claude LLM
- OpenAI embeddings + Ollama LLM
- Ollama embeddings + OpenAI LLM

**The only rule**: Query embeddings must match document embeddings.

---

## Deploy to HuggingFace Spaces

### Steps

1. **Create a HuggingFace account**
   - Sign up at huggingface.co

2. **Create a new Space**
   - Go to huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Gradio" as the SDK
   - Pick a name (e.g., `scipy-rag-assistant`)

---

3. **Prepare your files**

   ```
   app.py              # Your Gradio app (entry point)
   requirements.txt    # Dependencies
   src/                # Your modules
   chroma_db/          # Your vector store (or rebuild on startup)
   .env                # DON'T upload this!
   ```

4. **Add secrets (for API keys)**
   - In Space settings -> "Repository secrets"
   - Add `OPENAI_API_KEY` (and `ANTHROPIC_API_KEY` if needed)
   - These become environment variables

---

5. **Push to HuggingFace**

   ```bash
   # Option A: Use git
   git clone https://huggingface.co/spaces/YOUR_USERNAME/scipy-rag-assistant
   cp -r app.py src/ requirements.txt chroma_db/ scipy-rag-assistant/
   cd scipy-rag-assistant
   git add . && git commit -m "Initial deploy" && git push

   # Option B: Use the web UI to upload files
   ```

6. **Wait for build**
   - HuggingFace installs dependencies and launches
   - Check "Logs" tab if it fails

### ChromaDB Deployment Options

ChromaDB can be large. Either:

- Include a small `chroma_db/` in the repo
- Or rebuild on startup (slower cold start, but smaller repo)

---

## Module 4 Summary

### What We Built

1. **Evaluation Framework**: Retrieval + generation metrics
2. **Gradio App**: Interactive web interface
3. **Production Patterns**: Caching, rate limiting, updates

### Key Takeaways

| Concept              | Why It Matters                                |
| -------------------- | --------------------------------------------- |
| Retrieval metrics    | Precision, Recall, MRR measure search quality |
| Generation metrics   | Faithfulness, relevance, completeness         |
| LLM-as-judge         | Scalable evaluation without human reviewers   |
| Caching              | Avoid redundant API calls                     |
| Provider flexibility | Mix embeddings and LLMs freely                |

---

### The Complete Evaluation Flow

```
Build eval dataset (questions + expected functions)
     ↓
Run retrieval evaluation (Hit@K, MRR, Precision)
     ↓
Run generation evaluation (LLM-as-judge)
     ↓
Identify failure modes
     ↓
Iterate on chunking, prompts, or retrieval
```
