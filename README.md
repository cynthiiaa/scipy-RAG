# SciPy RAG Workshop

A comprehensive workshop teaching RAG (Retrieval-Augmented Generation) fundamentals and current practices, resulting in a SciPy code generation assistant.

## About the Author

**Cynthia Ukawu** | Lead Machine Learning Engineer

Cynthia independently designed and developed this project, including:

- The AI infrastructure and core functionality
- The structured workshop curriculum
- The supporting technical documentation

🔗 Read more technical deep-dives on AI/ML at [cynscode.com](https://cynscode.com).

## Workshop Overview

**Target Audience:** Engineers and scientists with Python experience

**What You'll Build:** A RAG-powered assistant that generates accurate SciPy code using up-to-date documentation

### ⚠️ Ollama vs Closed Source Models

If you have an OpenAI or Claude API key that you'd like to use for this workshop, Ollama is optional. **However,** if you don't have a proprietary API key, you are required to download Ollama beforehand. This ensures that you're able to follow along. Ollama is an open-source alternative and allows users to work with various open-source models. The two models we'll be using are `llama3.2` for the LLM and `nomic-embed-text` for embeddings.

## Learning Outcomes

By completing this workshop, you will:

- Understand RAG architecture and why it solves LLM limitations
- Build a document processing pipeline (scraping, chunking, embedding)
- Implement retrieval with ChromaDB vector database
- Create generation pipelines with OpenAI and/or Ollama
- Evaluate RAG system quality
- Deploy a Gradio web application

## Workshop Structure

| Module | Topic                             |
| ------ | --------------------------------- |
| 1      | RAG Fundamentals & Setup          |
| 2      | Building the SciPy Knowledge Base |
| 3      | RAG Pipeline & Generation         |
| 4      | Evaluation & Capstone App         |

Each module is **self-contained** and can be completed independently.

## Quick Start

### 1. Clone and Setup

```bash
# Navigate to the workshop directory
cd scipy-RAG

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your-key-here
```

### 3. (Optional) Setup Ollama for Local Models

```bash
# Install Ollama (macOS)
https://ollama.com/download/mac

# Install Ollama (Windows)
https://ollama.com/download/windows

# Install Ollama (Linux)
https://ollama.com/download/linux

# Start Ollama server
ollama serve

# Pull models
ollama pull llama3.2
ollama pull nomic-embed-text
```

#### (Optionally Optional) Pull More Models on Ollama

I'd recommend pulling these two additional models before the workshop because it takes a long time to download them 🐢 <br>
**You'll need at least 16GB of memory available.**

```bash
ollama pull mistral         # this pulls mistral:7b

ollama pull codellama       # this pulls codellama:7b
```

### 4. Start the Workshop

```bash
# Launch Jupyter
jupyter notebook notebooks/
```

Start with `01_rag_fundamentals.ipynb` and proceed in order.

## Module Descriptions

### Module 1: RAG Fundamentals & Setup

This module explains what RAG is, why it exists, and how embeddings turn text into searchable vectors. You'll set up ChromaDB and build a minimal RAG system from scratch. No frameworks, just the core concepts.

By the end, you'll have a working prototype that retrieves relevant documents and generates answers.

### Module 2: Building the SciPy Knowledge Base

This module walks through scraping SciPy documentation, cleaning it, and chunking it intelligently. Chunking matters more than most people expect. For example, if you split code in the wrong place this can lead to poor retrieval.

You'll populate a ChromaDB collection with embedded SciPy docs, ready for the next module.

### Module 3: RAG Pipeline & Generation

You'll build the full `SciPyRAG` class with:

- Query expansion and preprocessing
- Prompt templates tuned for code generation
- Support for both OpenAI and Ollama

The module also covers the embedding mismatch problem (why you can't mix embedding models between indexing and querying).

### Module 4: Evaluation & Capstone App

How do you know if your RAG system is any good? This module introduces retrieval metrics (precision, recall, MRR) and generation evaluation approaches.

You'll build a Gradio web app that ties everything together—a working SciPy assistant you can actually use.

~~Bonus: Advanced RAG Techniques~~

~~For those who want to go deeper. Covers HyDE (hypothetical document embeddings), query decomposition, hybrid search, and reranking. Also compares what the same RAG system looks like in LangChain vs LlamaIndex.~~

## Running the Gradio App

After completing Modules 1-4:

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Tech Stack

| Component    | Technology                                      |
| ------------ | ----------------------------------------------- |
| LLMs         | OpenAI GPT-4, Ollama (local)                    |
| Embeddings   | OpenAI text-embedding-3-small, nomic-embed-text |
| Vector DB    | ChromaDB                                        |
| Web Scraping | BeautifulSoup, requests                         |
| UI           | Gradio                                          |
| Evaluation   | Custom metrics, RAGAS                           |

## Requirements

- Python 3.10+
- OpenAI API key
- (Optional) Ollama for local models
- ~4GB disk space for vector store

## Troubleshooting

### "OPENAI_API_KEY not found"

Make sure you've created a `.env` file with your API key:

```bash
cp .env.example .env
# Edit .env and add your key
```

### "Ollama not available"

Ollama is optional. If you want to use it:

```bash
# Start the Ollama server
ollama serve

# In another terminal, pull a model
ollama pull llama3.2
```

### "Vector store is empty"

Run Module 2 notebook to populate the knowledge base before using Modules 3-4.

### ChromaDB errors

If you encounter ChromaDB issues, try resetting:

```python
# In a notebook
from vectorstore import VectorStore
store = VectorStore(collection_name="scipy_docs", persist_directory="./chroma_db")
store.reset_collection()
```

## Extending the Workshop

**Add more SciPy modules:**

```python
from scraper import SciPyDocsScraper
scraper = SciPyDocsScraper()
docs = scraper.scrape_all(modules=['signal', 'ndimage'])
```

**Try different embedding models:**

```python
from embeddings import get_embedding_provider
provider = get_embedding_provider("ollama", model="mxbai-embed-large")
```

**Deploy to HuggingFace Spaces:**

1. Create a new Space at huggingface.co
2. Upload `app.py`, `src/`, and `chroma_db/`
3. Add your OPENAI_API_KEY as a secret

## Resources

- [SciPy Documentation](https://docs.scipy.org)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Ollama](https://ollama.com)
- [Gradio Documentation](https://gradio.app/docs)

---

Built for the SciPy RAG Workshop | Happy Coding 💻
