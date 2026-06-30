class ConfigurationError(Exception):
    """Raised when a required configuration is missing."""
    pass


# --- Error messages ---

OPENAI_KEY_MISSING = """\
OpenAI API key not found.

Options:
  1. Set OPENAI_API_KEY in your .env file
  2. Use Claude instead: create_rag_system(llm_type='claude')
  3. Use Ollama (local): create_rag_system(llm_type='ollama')"""

OPENAI_KEY_MISSING_EMBEDDINGS = """\
OpenAI API key not found.

Options:
  1. Set OPENAI_API_KEY in your .env file
  2. Use local embeddings instead (no API key needed):
         create_rag_system(embedding_type='ollama')

If you only have a Claude API key, use:
     create_rag_system(llm_type='claude', embedding_type='ollama')"""

ANTHROPIC_KEY_MISSING = """\
Anthropic API key not found.

Options:
  1. Set ANTHROPIC_API_KEY in your .env file
  2. Use OpenAI instead: create_rag_system(llm_type='openai')
  3. Use Ollama (local): create_rag_system(llm_type='ollama')"""

ANTHROPIC_NOT_INSTALLED = """\
Anthropic package not installed.

Install it with:
    pip install anthropic"""

OLLAMA_NOT_INSTALLED = """\
Ollama package not installed.

Install it with:
    pip install ollama

You also need Ollama running locally:
    1. Install from https://ollama.com
    2. Run: ollama serve
    3. Pull a model: ollama pull {model}"""

OLLAMA_CONNECTION_ERROR = """\
Cannot connect to Ollama.

Make sure Ollama is running:
    ollama serve

Then pull your model:
    ollama pull {model}"""

OLLAMA_MODEL_NOT_FOUND = """\
Model '{model}' not found.

Pull it first:
    ollama pull {model}"""


# --- Helper functions ---

def require_openai_key(api_key: str, for_embeddings: bool = False) -> str:
    """Validate OpenAI API key, raise friendly error if missing."""
    import os
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        msg = OPENAI_KEY_MISSING_EMBEDDINGS if for_embeddings else OPENAI_KEY_MISSING
        raise ValueError(msg)
    return key


def require_anthropic_key(api_key: str) -> str:
    """Validate Anthropic API key, raise friendly error if missing."""
    import os
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(ANTHROPIC_KEY_MISSING)
    return key


def import_anthropic():
    """Import anthropic with friendly error."""
    try:
        import anthropic
        return anthropic
    except ImportError:
        raise ImportError(ANTHROPIC_NOT_INSTALLED)


def import_ollama(model: str = "llama3.2"):
    """Import ollama with friendly error."""
    try:
        import ollama
        return ollama
    except ImportError:
        raise ImportError(OLLAMA_NOT_INSTALLED.format(model=model))


def handle_ollama_error(e: Exception, model: str):
    """Convert Ollama exceptions to friendly errors."""
    error_msg = str(e).lower()
    if "connection" in error_msg or "refused" in error_msg:
        raise ConnectionError(OLLAMA_CONNECTION_ERROR.format(model=model)) from e
    if "not found" in error_msg or "pull" in error_msg:
        raise ValueError(OLLAMA_MODEL_NOT_FOUND.format(model=model)) from e
    raise
