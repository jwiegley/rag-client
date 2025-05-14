#!/usr/bin/env python
# pyright: reportUnknownVariableType=false

import atexit
import json
import logging
import os
import sys
import argparse
import readline
from pathlib import Path

from typed_argparse import TypedArgs
from xdg_base_dirs import xdg_config_home

from llama_index.core.base.response.schema import (
    StreamingResponse,
    Response,
)
from llama_index.core.storage.chat_store import SimpleChatStore

from rag import *
import api


# Define a class to hold the parsed arguments
class Args(TypedArgs):
    from_: str | None
    num_workers: int | None
    recursive: bool
    verbose: bool
    streaming: bool
    host: str
    port: int
    reload_server: bool
    config: str
    command: str
    args: list[str]


def parse_args(arguments: list[str] = sys.argv[1:]) -> Args:
    """
    Parse the command-line arguments.

    Args:
        arguments: A list of command-line arguments (default: sys.argv[1:])

    Returns:
        An Args object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="RAG Client")

    _ = parser.add_argument(
        "--from", dest="from_", type=str, help="Where to read files from (optional)"
    )
    _ = parser.add_argument(
        "--recursive",
        action="store_true",
        help="Whether to read directories recursively)",
    )
    _ = parser.add_argument(
        "--num-workers", "-j", type=int, help="Number of parallel jobs to use"
    )
    _ = parser.add_argument("--verbose", action="store_true", help="Verbose?")
    _ = parser.add_argument("--streaming", action="store_true", help="Streaming?")
    _ = parser.add_argument(
        "--host", type=str, default="localhost", help="Host for 'serve' command"
    )
    _ = parser.add_argument(
        "--port", type=int, default=7990, help="Port for 'serve' command"
    )
    _ = parser.add_argument(
        "--reload-server", action="store_true", help="Reload on source change?"
    )
    _ = parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Yaml config file, to set argument defaults",
    )
    _ = parser.add_argument("command")
    _ = parser.add_argument("args", nargs=argparse.REMAINDER)

    _ = parser.parse_known_args(arguments)

    return Args.from_argparse(parser.parse_args())


def search_command(rag: RAGWorkflow, retriever: BaseRetriever, query: str):
    """
    Execute a search command.

    Args:
        rag: The RAG workflow object.
        retriever: The retriever object.
        query: The search query.
    """
    # Retrieve nodes using the retriever and query
    nodes = rag.retrieve_nodes(retriever, query)
    # Print the retrieved nodes in JSON format
    print(json.dumps(nodes, indent=2))


def query_command(query_state: QueryState, query: QueryType):
    """
    Execute a query command.

    Args:
        query_state: The query state object.
        query: The query string.
    """
    # Execute the query using the query state
    response = query_state.query(query=query)
    # Handle different response types
    match response:
        case StreamingResponse():
            # Print the response tokens as they are generated
            for token in response.response_gen:
                token = clean_special_tokens(token)
                print(token, end="", flush=True)
            print()
        case Response():
            # Print the response
            print(response.response)
        case _:
            # Log an error for unhandled response types
            error(f"query_command cannot render response: {response}")


def chat_command(chat_state: ChatState, query: str, streaming: bool):
    """
    Execute a chat command.

    Args:
        chat_state: The chat state object.
        query: The query string.
        streaming: Whether to stream the response.
    """
    # Execute the chat using the chat state
    response = chat_state.chat(
        query=query,
        streaming=streaming,
    )
    # Handle streaming and non-streaming responses
    if streaming:
        # Print the response tokens as they are generated
        for token in response.response_gen:
            token = clean_special_tokens(token)
            print(token, end="", flush=True)
        print()
    else:
        # Print the response
        print(response.response)


def rag_client(
    rag: RAGWorkflow,
    retriever: BaseRetriever | None,
    args: Args,
):
    """
    Execute the RAG client.

    Args:
        rag: The RAG workflow object.
        retriever: The retriever object (optional).
        args: The parsed arguments.
    """
    # Handle different commands
    match args.command:
        case "search":
            # Check if a retriever is available
            if retriever is not None:
                search_command(rag, retriever, args.args[0])
            else:
                error("Search command requires a retriever")
        case "query":
            if rag.config.query is None:
                error("'query' command requires query engine to be configured")
            # Create a query state object
            llm = rag.realize_llm(rag.config.query.llm, verbose=args.verbose)
            query_state = QueryState(
                config=rag.config.query,
                llm=llm,
                retriever=retriever,
                streaming=args.streaming,
                verbose=args.verbose,
            )
            # Execute the query command
            query_command(query_state, query=args.args[0])
        case "chat":
            if rag.config.chat is None:
                error("'chat' command requires chat engine to be configured")

            # Path to the history file (change as needed)
            HISTFILE = xdg_config_home() / "rag-client" / "chat_history"

            # Load history if it exists
            try:
                readline.read_history_file(HISTFILE)
            except FileNotFoundError:
                pass

            # Optionally limit the history length
            readline.set_history_length(1000)

            # Save history on exit
            _ = atexit.register(readline.write_history_file, HISTFILE)

            # Optional: Enable tab completion (if you want)
            # readline.parse_and_bind("tab: complete")

            user = rag.config.chat.default_user or "user"

            # Create a chat store object
            chat_store_json = xdg_config_home() / "rag-client" / "chat_store.json"
            if rag.config.chat.keep_history:
                chat_store = SimpleChatStore.from_persist_path(str(chat_store_json))
            else:
                chat_store = SimpleChatStore()

            query_state: QueryState | None = None
            chat_state: ChatState | None = None

            # Enter a chat loop
            while True:
                query = input(f"\n{user}> ")
                if query.strip().lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    # Persist the chat store if required
                    if rag.config.chat.keep_history:
                        chat_store.persist(persist_path=str(chat_store_json))
                    break
                elif query.startswith("search "):
                    # Handle search queries within the chat loop
                    if retriever is not None:
                        search_command(rag, retriever, query[7:])
                elif query.startswith("query "):
                    if rag.config.query is None:
                        print("'query' command requires query engine to be configured")
                    else:
                        # Handle query commands within the chat loop
                        if query_state is None:
                            llm = rag.realize_llm(
                                rag.config.query.llm, verbose=args.verbose
                            )
                            query_state = QueryState(
                                config=rag.config.query,
                                llm=llm,
                                retriever=retriever,
                                streaming=args.streaming,
                                verbose=args.verbose,
                            )
                        query_command(query_state, query[6:])
                else:
                    # Handle chat queries
                    if chat_state is None:
                        llm = rag.realize_llm(
                            rag.config.chat.llm,
                            verbose=args.verbose,
                        )
                        chat_state = ChatState(
                            config=rag.config.chat,
                            llm=llm,
                            user="user",
                            retriever=retriever,
                            verbose=args.verbose,
                        )
                    chat_command(
                        chat_state,
                        query=query,
                        streaming=args.streaming,
                    )
        case _:
            # Log an error for unrecognized commands
            error(f"Command unrecognized: {args.command}")


def main(args: Args):
    """
    The main entry point.

    Args:
        args: The parsed arguments.
    """
    # Configure logging
    logging.basicConfig(
        stream=sys.stdout,
        encoding="utf-8",
        level=logging.INFO if args.verbose else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",  # Log message format
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("rag")

    # Initialize the RAG workflow and retriever
    (rag, retriever) = rag_initialize(
        logger=logger,
        config_path=Path(args.config),
        input_from=args.from_,
        num_workers=args.num_workers,
        recursive=args.recursive,
        verbose=args.verbose,
    )

    # Handle different commands
    match args.command:
        case "serve":
            # Set the API workflow
            api.workflow = rag
            # Start the API server
            api.start_api_server(
                host=args.host,
                port=args.port,
                reload=args.reload_server,
            )
        case "index":
            # Currently a no-op
            pass
        case _:
            # Execute the RAG client
            rag_client(rag, retriever, args)


if __name__ == "__main__":
    # Set an environment variable to avoid tokenizer parallelism issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Parse arguments and execute the main function
    main(parse_args())
