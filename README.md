# SciPy RAG Workshop

A comprehensive workshop teaching RAG (Retrieval-Augmented Generation) fundamentals and current practices, culminating in a SciPy code generation assistant.

## Workshop Overview

**Target Audience:** Engineers and scientists with Python experience

**What You'll Build:** A RAG-powered assistant that generates accurate SciPy code using up-to-date documentation

**Duration:** 6-8 hours (4 modules + 1 bonus)

## Learning Outcomes

By completing this workshop, you will:

- Understand RAG architecture and why it solves LLM limitations
- Build a document processing pipeline (scraping, chunking, embedding)
- Implement retrieval with ChromaDB vector database
- Create generation pipelines with OpenAI and Ollama
- Evaluate RAG system quality
- Deploy a Gradio web application

## Workshop Structure

| Module | Topic                             | Duration  |
| ------ | --------------------------------- | --------- |
| 1      | RAG Fundamentals & Setup          | 1.5-2 hrs |
| 2      | Building the SciPy Knowledge Base | 1.5-2 hrs |
| 3      | RAG Pipeline & Generation         | 1.5-2 hrs |
| 4      | Evaluation & Capstone App         | 1.5-2 hrs |
| Bonus  | Advanced RAG Techniques           | 1-1.5 hrs |

Each module is **self-contained** and can be completed independently.

## Quick Start

### 1. Clone and Setup

```bash
# Navigate to the workshop directory
cd scipy_rag

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
brew install ollama

# Another way to install Ollama (macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Link to download
https://ollama.com/download

# Start Ollama server
ollama serve

# Pull models
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Start the Workshop

```bash
# Launch Jupyter
jupyter notebook notebooks/
```

Start with `01_rag_fundamentals.ipynb` and proceed in order.

## Project Structure

```
scipy_rag/
├── notebooks/
│   ├── 01_rag_fundamentals.ipynb      # Module 1: RAG basics
│   ├── 02_scipy_knowledge_base.ipynb  # Module 2: Scraping & chunking
│   ├── 03_rag_pipeline.ipynb          # Module 3: Retrieval & generation
│   ├── 04_evaluation.ipynb            # Module 4: Evaluation & app
│   └── 05_advanced_rag.ipynb          # Bonus: Advanced techniques
├── src/
│   ├── scraper.py          # SciPy docs scraper
│   ├── chunker.py          # Document chunking utilities
│   ├── embeddings.py       # Embedding providers
│   ├── vectorstore.py      # ChromaDB wrapper
│   ├── retriever.py        # Retrieval logic
│   ├── generator.py        # LLM generation
│   └── rag.py              # Main RAG pipeline
├── data/
│   ├── raw/                # Scraped HTML (generated)
│   └── processed/          # Chunked documents
├── chroma_db/              # Vector store (generated)
├── evaluation/
│   └── test_questions.json # Evaluation dataset
├── app.py                  # Gradio web application
├── requirements.txt
└── .env.example
```

## Module Descriptions

### Module 1: RAG Fundamentals & Setup

**Learn:**

- What RAG is and why it matters
- Embeddings and vector similarity
- ChromaDB basics

**Build:**

- Mini RAG system with sample data
- Embedding visualization

### Module 2: Building the SciPy Knowledge Base

**Learn:**

- Web scraping best practices
- Document chunking strategies
- Code-aware text processing

**Build:**

- SciPy documentation scraper
- Production-ready vector store

### Module 3: RAG Pipeline & Generation

**Learn:**

- Query preprocessing techniques
- Prompt engineering for code generation
- Multi-provider LLM integration

**Build:**

- Complete `SciPyRAG` class
- OpenAI + Ollama support

### Module 4: Evaluation & Capstone App

**Learn:**

- Retrieval metrics (precision, recall, MRR)
- Generation evaluation
- Production considerations

**Build:**

- Evaluation framework
- Gradio web application

### Bonus: Advanced RAG Techniques

**Learn:**

- HyDE (Hypothetical Document Embeddings)
- Query decomposition
- Hybrid search (dense + sparse)
- Reranking

**Compare:**

- LangChain implementation
- LlamaIndex implementation

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

## License

MIT License - Feel free to use this workshop material for learning and teaching.

Built for the SciPy RAG Workshop | Happy Coding 💻
