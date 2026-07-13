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

# Module 3: RAG Pipeline & Generation

**Prerequisites**: ChromaDB populated with SciPy docs

---

## 1. Module Goals

By the end of this module, you will:

1. Build a complete **retrieval pipeline** with preprocessing
2. Learn **prompt engineering** for code generation
3. Integrate **multiple LLM providers** (OpenAI, Claude, Ollama)
4. Assemble the full **SciPyRAG** system

---

## 2. Understanding the RAG Architecture

The retrieval and generation are **completely separate**:

| Step           | Component                                | Your Setup                      |
| -------------- | ---------------------------------------- | ------------------------------- |
| **Retrieval**  | Embedding model → ChromaDB search        | text-embedding-3-small (OpenAI) |
| **Generation** | LLM generates answer from retrieved docs | gpt-4o-mini (or Ollama)         |

---

### The Flow

```
Your question
     ↓
Embed query (OpenAI embeddings)
     ↓
Search ChromaDB scipy_docs → retrieve relevant chunks
     ↓
Format context + question
     ↓
Send to LLM (gpt-4o-mini) → generate answer
```

‼️ The query and the documents need to be embedded with the same embedding model so their vectors live in the same vector space.

---

### The Embedding Mismatch Problem

Switching between GPT-4o-mini and Ollama **only changes who generates the answer**. The retrieval from ChromaDB always uses the same embedding model that was used to index the documents.

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

The LLM does not answer from the embeddings directly. The embeddings are used to find the relevant text. The text is then sent to the model as context.

So you can mix:

- OpenAI embeddings + Claude LLM ✓
- OpenAI embeddings + Ollama LLM ✓
- Ollama embeddings + OpenAI LLM ✓

**However,** query embeddings must match document embeddings.

---

#### How Embeddings Work (When Matched)

```
Indexing (storing documents):
"scipy.optimize.minimize..." → Embedding Model → [0.12, -0.34, 0.56, ...] (1536 floats)
                                                        ↓
                                                 ChromaDB stores this vector

Querying (searching):
"How do I minimize a function?" → Embedding Model → [0.08, -0.29, 0.61, ...] (1536 floats)
                                                        ↓
                                                Compare to stored vectors
                                                (cosine similarity)
                                                        ↓
                                                Return closest matches
```

---

#### Mismatch Example

Your collection was indexed with the default embedding model (384 dimensions), but you queried with OpenAI (1536 dimensions):

```
Stored vectors:    [0.12, -0.34, 0.56, ...384 values]
Query vector:      [0.08, -0.29, 0.61, ...1536 values]
                          ↓
        Can't compute similarity because they have different sizes!
```

Cosine similarity requires both vectors to have the same number of dimensions.

---

#### Analogy: Coordinate Reference Systems

If you've worked with geospatial data you've probably worked with CRS. EPSG:4326 stores location as longitude/latitude angles on the globe. EPSG:3857 projects the globe onto a flat map and stores location as x/y coordinates in meters. The numbers may look similar, but comparing them directly would be meaningless. Embeddings are the same way: each model creates its own learned coordinate system.

#### Analogy: Tokenizers

Just like you should not assume two tokenizers split text the same way, you should not assume two embedding models place text in the same vector space. The vectors need to come from the same embedding model so the similarity scores are meaningful.

Embedding models are like tokenizers in the sense that each model has its own way of representing text. A tokenizer decides how text gets broken into tokens. An embedding model decides how text gets mapped into vector space. If you change the model, you change the representation.

**⭐️ Always use the same embedding model for indexing and querying.**

---

## 3. Query Preprocessing

### Why Preprocess?

User queries are often:

- **Vague**
- **Missing context**
- **Not code-focused**

### Preprocessing Techniques

| Technique         | Description                      | Example                                          |
| ----------------- | -------------------------------- | ------------------------------------------------ |
| **Expansion**     | Add synonyms and related terms   | "fit" → "curve fit, regression, least squares"   |
| **Rephrasing**    | Make queries more code-focused   | "minimize something" → "scipy minimize function" |
| **Decomposition** | Break complex queries into parts | "fit and plot" → ["fit curve", "plot results"]   |

---

## 4. Query Expansion

This function makes the user’s search query a little more flexible. Instead of searching only for the exact words the user typed, it creates a few related versions of the query using common SciPy terms and synonyms.

```python
def expand_query(query: str) -> list[str]:
    """Expand a query with synonyms and related terms."""
    variations = [query]

    synonyms = {
        "fit": ["curve fit", "regression", "least squares"],
        "minimize": ["optimize", "find minimum"],
        "integrate": ["integral", "quadrature"],
        "solve": ["find roots", "equation solver"],
        "interpolate": ["spline", "interpolation"],
    }

    for keyword, expansions in synonyms.items():
        if keyword in query.lower():
            for expansion in expansions:
                variations.append(f"{query} {expansion}")

    return variations
```

---

### Example

```python
>>> expand_query("fit a curve")
['fit a curve', 'fit a curve curve fit', 'fit a curve regression', 'fit a curve least squares']
```

The retriever can search with multiple versions of the same question. Sometimes, users may describe a problem one way, while the documentation describes it another way. Query expansion helps bridge that gap.

**With query expansion, we're adding related terms so the vector database has more chances to find the right chunks.**

---

## 5. Multi-Query Retrieval

This function takes several versions of the user’s question, searches the vector database with each one, and combines the results into one deduplicated list.

```python
def retrieve_multi_query(queries: list[str], top_k: int = 3):
    """Retrieve documents using multiple query variations."""
    all_results = []
    seen_ids = set()

    for query in queries:
        results = collection.query(query_texts=[query], n_results=top_k)

        for doc_id, doc, meta in zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0]
        ):
            if doc_id not in seen_ids:
                all_results.append((doc_id, doc, meta))
                seen_ids.add(doc_id)

    return all_results
```

---

This pairs well with `expand_query()`. First, you create query variations, then `retrieve_multi_query()` searches ChromaDB with each variation.

**Instead of betting everything on one phrasing of the question, we search with multiple related phrasings and collect the unique results. This improves recall.**

---

## 6. Context Formatting

This function takes the chunks we retrieved from the vector database and turns them into one clean context block that we can send to the LLM.

```python
def format_context(results: list, max_chars: int = 4000) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    context_parts = []
    total_chars = 0

    for result in results:
        header = f"[{result.metadata['function_name']}]\n"
        text = f"{header}{result.text}\n\n---\n\n"

        if total_chars + len(text) > max_chars:
            break

        context_parts.append(text)
        total_chars += len(text)

    return "".join(context_parts)
```

---

In RAG, retrieval gives us separate chunks:

```bash
chunk 1: curve_fit documentation
chunk 2: least squares documentation
chunk 3: optimization documentation
```

But the LLM needs them formatted into a prompt-friendly string.
**`max_chars` prevents the prompt from getting too large.**

### Why Limit Context Size?

- LLMs have context windows (token limits)
- More context isn't always better (irrelevant chunks add noise)
- Cost scales with tokens (for paid APIs)

---

## 7. Prompt Engineering

### The System Prompt

```python
SYSTEM_PROMPT = """You are an expert SciPy programming assistant.

Guidelines:
1. Always provide working, runnable code examples
2. Include necessary imports at the top
3. Add brief comments explaining key steps
4. If context doesn't fully answer, say so
5. Prefer simple, readable solutions

When generating code:
- Use numpy and scipy best practices
- Handle edge cases appropriately
- Suggest reasonable parameter values"""
```

---

### The User Prompt Template

```python
USER_PROMPT = """Based on the following SciPy documentation:

---
{context}
---

User Question: {question}

Provide a clear answer with working code examples.
If the documentation doesn't contain enough information,
mention what additional details might be needed."""
```

### Key Elements

- Clear separation of context and question
- Explicit instructions for code
- Handling incomplete information gracefully

---

## 8. Different Prompt Styles

### Basic

```
Context: {context}
Question: {question}
```

Simple but may produce inconsistent output format.

---

### Structured

```markdown
## Documentation

{context}

## Question

{question}

## Instructions

1. Provide direct answer
2. Include code example
3. Explain key parameters
```

Better control over output structure.

---

### Few-Shot

Few-shot prompting is when we include example Q&A pairs in the prompt so the model can copy the style, structure, and level of detail we want.

````python
EXAMPLE = """
Question: How do I compute a definite integral?
Answer: Use scipy.integrate.quad():

```python
from scipy.integrate import quad

result, error = quad(lambda x: x**2, 0, 1)
print(f"Integral: {result}")  # 0.333...
```
````

Most reliable for consistent formatting.

---

## 9. LLM Provider: OpenAI

```python
from openai import OpenAI

class OpenAIGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content
```

---

Temperature controls how much variation or randomness we allow in the model’s response. For RAG/coding assistants, you usually want a lower temperature, because you want the model to produce reliable code.

### Temperature Setting

- `temperature=0`: Deterministic, consistent outputs
- `temperature=0.3`: Slight variation, good for code
- `temperature=0.7+`: More creative, riskier for code

---

## 10. LLM Provider: Ollama (Local)

```python
import ollama

class OllamaGenerator:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.model,
            messages=messages
        )
        return response['message']['content']
```

---

### Benefits of Local Models

- **No API costs**: Free after download
- **Works offline**: No internet required
- **Data privacy**: Code never leaves your machine
- **Low latency**: No network round-trip

---

## 11. Streaming Responses

Better UX with token-by-token streaming:

```python
def generate_stream(self, prompt: str):
    """Stream response tokens as they're generated."""
    stream = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Usage
for token in generator.generate_stream("Explain curve fitting"):
    print(token, end="", flush=True)
```

Streaming makes the system feel more responsive, especially for longer answers.

---

## 12. The Complete SciPyRAG Class

```python
class SciPyRAG:
    def __init__(self, vector_store, llm_provider):
        self.vector_store = vector_store
        self.retriever = Retriever(vector_store)
        self.llm = llm_provider

    def query(self, question: str) -> RAGResponse:
        # 1. Retrieve relevant chunks
        results = self.retriever.retrieve_with_context(question)

        # 2. Format context for LLM
        context = self.retriever.format_context(results)

        # 3. Create prompts
        system, user = create_scipy_prompt(question, context)

        # 4. Generate answer
        answer = self.llm.generate(user, system_prompt=system)

        return RAGResponse(question, answer, results, context)

    def switch_llm(self, provider: str, model: str):
        """Switch to a different LLM provider."""
        if provider == "openai":
            self.llm = OpenAIGenerator(model)
        elif provider == "ollama":
            self.llm = OllamaGenerator(model)
        elif provider == "claude":
            self.llm = ClaudeGenerator(model)
```

---

## 13. Using the RAG System

```python
# Initialize
rag = SciPyRAG(
    vector_store=vector_store,
    llm_provider=OpenAIGenerator()
)

# Query
response = rag.query("How do I fit an exponential curve to my data?")

print(f"Answer: {response.answer}")
print(f"Sources: {[s.metadata['function_name'] for s in response.sources]}")

# Switch to local model
rag.switch_llm("ollama", "llama3.2")
response = rag.query("What is scipy.integrate.quad?")
```

---

## Module 3 Summary

### What We Learned/Built

1. **Query Preprocessing**: Expansion, multi-query retrieval
2. **Context Formatting**: Structured context for LLMs
3. **Prompt Engineering**: System + user prompt templates
4. **LLM Providers**: OpenAI + Ollama with streaming
5. **SciPyRAG**: Complete pipeline class

---

### Key Takeaways

| Concept               | Why It Matters                                |
| --------------------- | --------------------------------------------- |
| Query preprocessing   | Improves retrieval recall                     |
| Prompt engineering    | Significantly affects output quality          |
| Multiple providers    | Flexibility (cloud vs local, cost vs quality) |
| Streaming             | Better user experience                        |
| Embedding consistency | Queries must use same model as indexing       |

---

### The Complete Flow

```
User question
     ↓
Query expansion (synonyms)
     ↓
Embed query (OpenAI)
     ↓
Search ChromaDB → retrieve top-k chunks
     ↓
Format context (with headers, truncation)
     ↓
Build prompt (system + user)
     ↓
Generate answer (GPT-4o-mini / Ollama / Claude)
     ↓
Return answer + sources
```

---

## Quick Reference

### Query Expansion Template

```python
def expand_query(query: str) -> list[str]:
    variations = [query]
    synonyms = {"fit": ["curve fit", "regression"], ...}
    for keyword, expansions in synonyms.items():
        if keyword in query.lower():
            variations.extend([f"{query} {e}" for e in expansions])
    return variations
```

### Basic RAG Query

```python
# Retrieve
results = collection.query(query_texts=[question], n_results=3)
context = "\n".join(results['documents'][0])

# Generate
response = llm.generate(f"Context: {context}\n\nQuestion: {question}")
```

---

### Switching LLM Providers

```python
# OpenAI
rag.switch_llm("openai", "gpt-4o-mini")

# Ollama (local)
rag.switch_llm("ollama", "llama3.2")

# Claude
rag.switch_llm("claude", "claude-sonnet-4-20250514")
```

### Prompt Template

```python
SYSTEM = "You are a SciPy expert. Provide working code with imports."

USER = f"""Documentation:
{context}

Question: {question}

Provide a clear answer with code examples."""
```
