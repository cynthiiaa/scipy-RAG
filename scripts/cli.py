"""
SciPy RAG Assistant - Command Line Interface

Usage:
    python cli.py "How do I minimize a function?"
    python cli.py --file mycode.py "How can I improve this?"
    python cli.py --file mycode.py --analyze
    python cli.py --chat

Examples:
    # Ask a question
    python cli.py "What's the best way to fit a curve?"

    # Analyze your code for SciPy improvements
    python cli.py --file optimization.py --analyze

    # Ask about specific code
    python cli.py --file signal_processing.py "Is there a better filter to use here?"

    # Interactive chat mode
    python cli.py --chat
"""

import argparse
import sys
import threading
import itertools
import time
from pathlib import Path


class Spinner:
    """Simple CLI spinner for long-running operations."""
    def __init__(self, message="Thinking"):
        self.message = message
        self.running = False
        self.thread = None

    def _spin(self):
        for char in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if not self.running:
                break
            print(f"\r{self.message} {char}", end="", flush=True)
            time.sleep(0.1)
        print("\r" + " " * (len(self.message) + 2) + "\r", end="", flush=True)

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.running = False
        self.thread.join()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from rag import create_rag_system


def load_rag():
    """Initialize the RAG system."""
    chroma_path = Path(__file__).parent.parent / "chroma_db"
    if not chroma_path.exists():
        print("Error: ChromaDB not found. Run notebook 02 first to build the index.")
        sys.exit(1)
    return create_rag_system(chroma_path=str(chroma_path))


def read_code_file(filepath: str) -> str:
    """Read a code file and return its contents."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    return path.read_text()


def analyze_code(rag, code: str, filename: str) -> str:
    """Analyze code for potential SciPy improvements."""
    prompt = f"""I have the following Python code from '{filename}':

```python
{code}
```

Please analyze this code and suggest:
1. Are there any SciPy functions that could improve or replace parts of this code?
2. Any performance improvements using SciPy/NumPy?
3. Any potential bugs or issues with the numerical computations?
4. Better practices for scientific computing?

Focus on practical, actionable suggestions."""

    with Spinner("Analyzing..."):
        response = rag.query(prompt)
    return response.answer


def ask_about_code(rag, code: str, filename: str, question: str) -> str:
    """Ask a question about specific code."""
    prompt = f"""I have the following Python code from '{filename}':

```python
{code}
```

My question: {question}"""

    with Spinner("Tinkering..."):
        response = rag.query(prompt)
    return response.answer


def ask_question(rag, question: str, show_spinner: bool = True) -> str:
    """Ask a general SciPy question."""
    if show_spinner:
        with Spinner("Waddling..."):
            response = rag.query(question)
    else:
        response = rag.query(question)
    return response.answer


def interactive_chat(rag):
    """Run interactive chat mode."""
    print("\nSciPy RAG Assistant")
    print("=" * 40)
    print("Ask questions about SciPy. Type 'quit' to exit.")
    print("Type 'file:path/to/code.py' to load a file for context.\n")

    current_code = None
    current_filename = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Check for file loading command
        if user_input.lower().startswith("file:"):
            filepath = user_input[5:].strip()
            try:
                current_code = read_code_file(filepath)
                current_filename = Path(filepath).name
                lines = len(current_code.splitlines())
                print(f"Loaded {current_filename} ({lines} lines). You can now ask questions about it.\n")
            except SystemExit:
                pass
            continue

        # Check for analyze command
        if user_input.lower() == "analyze" and current_code:
            answer = analyze_code(rag, current_code, current_filename)
            print(f"\nAssistant: {answer}\n")
            continue

        # Regular question (with or without code context)
        if current_code and any(word in user_input.lower() for word in ["this", "code", "it", "here", "my"]):
            answer = ask_about_code(rag, current_code, current_filename, user_input)
        else:
            answer = ask_question(rag, user_input)

        print(f"\nAssistant: {answer}\n")


def main():
    parser = argparse.ArgumentParser(
        description="SciPy RAG Assistant - Get help with SciPy from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask about SciPy"
    )
    parser.add_argument(
        "--file", "-f",
        help="Path to a Python file to analyze or ask about"
    )
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Analyze the file for SciPy improvements"
    )
    parser.add_argument(
        "--chat", "-c",
        action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["openai", "claude", "ollama"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.chat and not args.question and not (args.file and args.analyze):
        parser.print_help()
        sys.exit(1)

    # Load RAG system
    print("Loading SciPy RAG Assistant...", file=sys.stderr)
    rag = load_rag()

    # Switch model if needed
    if args.model == "claude":
        rag.switch_llm("claude", "claude-sonnet-4-20250514")
    elif args.model == "ollama":
        rag.switch_llm("ollama", "llama3.2")

    print(f"Ready! ({rag.vector_store.count()} docs indexed)\n", file=sys.stderr)

    # Handle different modes
    if args.chat:
        interactive_chat(rag)

    elif args.file and args.analyze:
        code = read_code_file(args.file)
        filename = Path(args.file).name
        print(f"Analyzing {filename}...\n")
        answer = analyze_code(rag, code, filename)
        print(answer)

    elif args.file and args.question:
        code = read_code_file(args.file)
        filename = Path(args.file).name
        answer = ask_about_code(rag, code, filename, args.question)
        print(answer)

    elif args.question:
        answer = ask_question(rag, args.question)
        print(answer)


if __name__ == "__main__":
    main()
