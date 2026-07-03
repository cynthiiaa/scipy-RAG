"""
SciPy RAG Assistant - Gradio Application (Standalone Script)

A web interface for the SciPy RAG system that provides accurate,
documentation-grounded answers to SciPy questions.

Run with: python app.py
Or: gradio app.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG system
from rag import SciPyRAG, create_rag_system


# Initialize RAG system
print("Initializing SciPy RAG system...")
chroma_path = Path(__file__).parent.parent / 'chroma_db'
rag = create_rag_system(chroma_path=str(chroma_path))
print(f"Loaded {rag.vector_store.count()} documents")


def query_scipy(
    question: str,
    model: str = "GPT-4o-mini",
    show_sources: bool = True,
    show_context: bool = False
):
    """
    Query the SciPy RAG system.

    Args:
        question: User's question
        model: LLM to use
        show_sources: Whether to show source documents
        show_context: Whether to show retrieved context

    Returns:
        Tuple of (answer, sources, context)
    """
    if not question.strip():
        return "Please enter a question about SciPy.", "", ""

    # Switch model if needed
    try:
        if model == "GPT-4o-mini":
            rag.switch_llm("openai", "gpt-4o-mini")
        elif model == "GPT-4o":
            rag.switch_llm("openai", "gpt-4o")
        elif model == "Ollama (llama3.2)":
            rag.switch_llm("ollama", "llama3.2")
        elif model == "Ollama (codellama)":
            rag.switch_llm("ollama", "codellama")
    except Exception as e:
        return f"Error switching model: {e}. Using default.", "", ""

    # Query RAG system
    try:
        response = rag.query(question)
    except Exception as e:
        return f"Error generating response: {e}", "", ""

    # Format sources
    sources_md = ""
    if show_sources and response.sources:
        sources_md = "### Sources\n\n"
        for i, s in enumerate(response.sources[:5], 1):
            func = s.metadata.get('function_name', 'Unknown')
            module = s.metadata.get('module', '')
            chunk_type = s.metadata.get('chunk_type', '')
            score = s.score
            sources_md += f"{i}. **{module}.{func}** ({chunk_type}) - Relevance: {score:.2%}\n"

    # Format context
    context_md = ""
    if show_context and response.context_used:
        context_md = f"### Retrieved Context\n\n```\n{response.context_used[:3000]}\n```"

    return response.answer, sources_md, context_md


def create_app():
    """Create the Gradio application."""

    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .answer-box {
        min-height: 200px;
    }
    """

    with gr.Blocks(
        title="SciPy RAG Assistant",
        css=css,
        theme=gr.themes.Soft()
    ) as app:

        # Header
        gr.Markdown(
            """
            # SciPy RAG Assistant

            Ask questions about SciPy and get accurate, documentation-grounded answers.
            Powered by RAG (Retrieval-Augmented Generation) for up-to-date responses.
            """
        )

        # Main interface
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="How do I fit an exponential curve to my data?",
                    lines=3,
                    max_lines=5
                )

                with gr.Row():
                    submit_btn = gr.Button("Ask", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

            with gr.Column(scale=1):
                model_select = gr.Dropdown(
                    choices=[
                        "GPT-4o-mini",
                        "GPT-4o",
                        "Ollama (llama3.2)",
                        "Ollama (codellama)"
                    ],
                    value="GPT-4o-mini",
                    label="Model"
                )
                show_sources = gr.Checkbox(label="Show Sources", value=True)
                show_context = gr.Checkbox(label="Show Context", value=False)

        # Output area
        with gr.Row():
            with gr.Column(scale=2):
                answer_output = gr.Markdown(
                    label="Answer",
                    elem_classes=["answer-box"]
                )
            with gr.Column(scale=1):
                sources_output = gr.Markdown(label="Sources")

        # Context (collapsible)
        with gr.Accordion("Retrieved Context", open=False):
            context_output = gr.Markdown()

        # Examples
        gr.Markdown("### Example Questions")
        examples = gr.Examples(
            examples=[
                ["How do I minimize a function in SciPy?"],
                ["What's the best way to fit an exponential curve to data?"],
                ["How can I calculate the definite integral of a function?"],
                ["I need to solve a system of linear equations Ax = b"],
                ["How do I design a Butterworth lowpass filter?"],
                ["What function should I use for 1D interpolation?"],
                ["How can I compute the FFT of a signal?"],
                ["I want to calculate distances between points in two arrays"],
            ],
            inputs=question_input,
            label="Click to try"
        )

        # Footer
        gr.Markdown(
            """
            ---
            Built with RAG for accurate, up-to-date SciPy assistance.
            [Workshop Materials](https://github.com/your-repo) |
            [SciPy Documentation](https://docs.scipy.org)
            """
        )

        # Event handlers
        def submit_query(question, model, sources, context):
            return query_scipy(question, model, sources, context)

        def clear_all():
            return "", "", "", ""

        submit_btn.click(
            fn=submit_query,
            inputs=[question_input, model_select, show_sources, show_context],
            outputs=[answer_output, sources_output, context_output]
        )

        question_input.submit(
            fn=submit_query,
            inputs=[question_input, model_select, show_sources, show_context],
            outputs=[answer_output, sources_output, context_output]
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[question_input, answer_output, sources_output, context_output]
        )

    return app


# Create and launch app
app = create_app()

if __name__ == "__main__":
    # Launch options
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set True for public URL
        show_error=True
    )
