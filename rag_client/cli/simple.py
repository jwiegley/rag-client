"""Simplified CLI for RAG client with zero-config defaults.

This module provides a streamlined command-line interface that works
without requiring YAML configuration files for common use cases.

Usage:
    rag index ./docs             # Index a directory
    rag search "query"           # Search indexed documents
    rag query "question"         # Ask a question with AI response
    rag chat                     # Interactive chat mode
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from ..config.models import (
    ChatConfig,
    Config,
    HuggingFaceEmbeddingConfig,
    OllamaConfig,
    QueryConfig,
    RetrievalConfig,
    RetrieverQueryEngineConfig,
    SentenceSplitterConfig,
    SimpleContextChatEngineConfig,
)
from ..core.workflow import RAGWorkflow
from ..exceptions import ConfigurationError, RAGClientError
from ..utils.logging import get_logger, setup_logging


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TOP_K = 5


def detect_ollama() -> bool:
    """Check if Ollama is available."""
    try:
        import urllib.request

        req = urllib.request.Request(f"{DEFAULT_OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def detect_available_model() -> Optional[str]:
    """Detect available Ollama model."""
    try:
        import urllib.request
        import json as j

        req = urllib.request.Request(f"{DEFAULT_OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = j.loads(resp.read().decode())
            models = data.get("models", [])
            if models:
                preferred = [
                    "llama3.2",
                    "llama3.1",
                    "llama3",
                    "llama2",
                    "mistral",
                    "qwen",
                ]
                for pref in preferred:
                    for m in models:
                        if pref in m.get("name", "").lower():
                            return m["name"].split(":")[0]
                return models[0]["name"].split(":")[0]
    except Exception:
        pass
    return None


def create_default_config(
    model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> Config:
    """Create a default configuration for common use cases.

    Args:
        model: LLM model name (auto-detected if None)
        embedding_model: Embedding model name
        ollama_url: Ollama server URL
        top_k: Number of results to retrieve

    Returns:
        Config object with sensible defaults
    """
    llm_model = model or detect_available_model() or DEFAULT_LLM_MODEL
    embed_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    ollama = ollama_url or DEFAULT_OLLAMA_URL

    llm_config = OllamaConfig(
        model=llm_model,
        base_url=ollama,
        temperature=0.7,
        context_window=8192,
    )

    return Config(
        retrieval=RetrievalConfig(
            embedding=HuggingFaceEmbeddingConfig(
                model_name=embed_model,
                normalize=True,
            ),
            splitter=SentenceSplitterConfig(
                chunk_size=512,
                chunk_overlap=50,
            ),
        ),
        query=QueryConfig(
            llm=llm_config,
            engine=RetrieverQueryEngineConfig(),
        ),
        chat=ChatConfig(
            llm=llm_config,
            engine=SimpleContextChatEngineConfig(
                context_window=8192,
            ),
            keep_history=True,
        ),
    )


def simple_index(
    paths: List[str],
    recursive: bool = True,
    verbose: bool = False,
    force: bool = False,
    config: Optional[Config] = None,
) -> None:
    """Index documents with progress feedback.

    Args:
        paths: Paths to index (files or directories)
        recursive: Whether to recurse into directories
        verbose: Show verbose output
        force: Force re-indexing
        config: Optional custom configuration
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    cfg = config or create_default_config()
    logger = get_logger("rag")

    from ..utils.helpers import read_files

    all_files: List[Path] = []
    for path in paths:
        try:
            files = read_files(path, recursive)
            all_files.extend(files)
        except RAGClientError as e:
            console.print(f"[yellow]Warning:[/yellow] {e}")

    if not all_files:
        console.print("[red]No files found to index[/red]")
        return

    console.print(f"Found [green]{len(all_files)}[/green] files to index")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing documents...", total=None)

        rag = RAGWorkflow(logger, cfg)
        _retriever = rag.load_retriever(
            input_files=all_files,
            index_files=force,
            verbose=verbose,
        )

        progress.update(task, description="[green]Indexing complete![/green]")

    console.print(f"[green]Successfully indexed {len(all_files)} documents[/green]")


def simple_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = False,
    config: Optional[Config] = None,
) -> List[dict]:
    """Search indexed documents.

    Args:
        query: Search query
        top_k: Number of results
        verbose: Show verbose output
        config: Optional custom configuration

    Returns:
        List of search results with text and metadata
    """
    cfg = config or create_default_config(top_k=top_k)
    logger = get_logger("rag")

    rag = RAGWorkflow(logger, cfg)
    retriever = rag.load_retriever(
        input_files=None,
        top_k=top_k,
        verbose=verbose,
    )

    if retriever is None:
        raise ConfigurationError("No index found. Run 'rag index' first.")

    results = rag.retrieve_nodes(retriever, query)
    return results


def simple_query(
    question: str,
    streaming: bool = True,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = False,
    config: Optional[Config] = None,
) -> str:
    """Ask a question and get an AI-generated answer.

    Args:
        question: Question to ask
        streaming: Stream the response
        top_k: Number of context documents
        verbose: Show verbose output
        config: Optional custom configuration

    Returns:
        AI-generated answer
    """
    from ..core.models import QueryState

    cfg = config or create_default_config(top_k=top_k)
    logger = get_logger("rag")

    rag = RAGWorkflow(logger, cfg)
    retriever = rag.load_retriever(
        input_files=None,
        top_k=top_k,
        verbose=verbose,
    )

    if cfg.query is None:
        raise ConfigurationError("Query configuration missing")

    llm = rag.realize_llm(cfg.query.llm, verbose=verbose)
    query_state = QueryState(
        config=cfg.query,
        llm=llm,
        retriever=retriever,
        streaming=streaming,
        verbose=verbose,
    )

    response = query_state.query(question)

    if hasattr(response, "response_gen") and streaming:
        from ..utils.helpers import clean_special_tokens

        output = []
        for token in response.response_gen:
            token = clean_special_tokens(token)
            print(token, end="", flush=True)
            output.append(token)
        print()
        return "".join(output)
    else:
        return response.response


def simple_chat(
    streaming: bool = True,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = False,
    config: Optional[Config] = None,
) -> None:
    """Interactive chat mode.

    Args:
        streaming: Stream responses
        top_k: Number of context documents
        verbose: Show verbose output
        config: Optional custom configuration
    """
    from ..core.models import ChatState
    from ..utils.helpers import clean_special_tokens

    cfg = config or create_default_config(top_k=top_k)
    logger = get_logger("rag")

    rag = RAGWorkflow(logger, cfg)
    retriever = rag.load_retriever(
        input_files=None,
        top_k=top_k,
        verbose=verbose,
    )

    if cfg.chat is None:
        raise ConfigurationError("Chat configuration missing")

    llm = rag.realize_llm(cfg.chat.llm, verbose=verbose)
    chat_state = ChatState(
        config=cfg.chat,
        llm=llm,
        user="user",
        retriever=retriever,
        verbose=verbose,
    )

    print("RAG Chat (type 'exit' to quit)")
    print("-" * 40)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)

        response = chat_state.chat(query=query, streaming=streaming)

        if streaming:
            for token in response.response_gen:
                token = clean_special_tokens(token)
                print(token, end="", flush=True)
            print()
        else:
            print(response.response)


def main() -> None:
    """Main entry point for simplified CLI."""
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Simple RAG CLI - Index and query your documents",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show verbose output",
    )
    parser.add_argument(
        "--model",
        type=str,
        help=f"LLM model name (default: auto-detect or {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML config file (overrides auto-config)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("paths", nargs="+", help="Paths to index")
    index_parser.add_argument(
        "--no-recursive",
        "-n",
        action="store_true",
        help="Don't recurse into directories",
    )
    index_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-indexing",
    )

    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming",
    )

    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming",
    )

    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "WARNING"
    setup_logging(level=log_level, colored=True)

    config = None
    if args.config:
        config = RAGWorkflow.load_config(Path(args.config))
    else:
        config = create_default_config(
            model=args.model,
            embedding_model=args.embedding_model,
            ollama_url=args.ollama_url,
            top_k=args.top_k,
        )

    try:
        if args.command == "index":
            simple_index(
                paths=args.paths,
                recursive=not args.no_recursive,
                verbose=args.verbose,
                force=args.force,
                config=config,
            )

        elif args.command == "search":
            results = simple_search(
                query=args.query,
                top_k=args.top_k,
                verbose=args.verbose,
                config=config,
            )

            if args.json:
                print(json.dumps(results, indent=2))
            else:
                for i, result in enumerate(results, 1):
                    print(f"\n--- Result {i} ---")
                    print(result.get("text", "")[:500])
                    if result.get("metadata"):
                        print(
                            f"\nSource: {result['metadata'].get('file_name', 'Unknown')}"
                        )

        elif args.command == "query":
            simple_query(
                question=args.question,
                streaming=not args.no_stream,
                top_k=args.top_k,
                verbose=args.verbose,
                config=config,
            )

        elif args.command == "chat":
            simple_chat(
                streaming=not args.no_stream,
                top_k=args.top_k,
                verbose=args.verbose,
                config=config,
            )

        else:
            parser.print_help()

    except RAGClientError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
