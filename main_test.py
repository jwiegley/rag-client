#!/usr/bin/env python
# pyright: reportUnknownVariableType=false

import argparse
import atexit
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, NoReturn, Optional

import readline
from llama_index.core.base.response.schema import (
    Response,
    StreamingResponse,
)
from llama_index.core.storage.chat_store import SimpleChatStore
from typed_argparse import TypedArgs
from xdg_base_dirs import xdg_config_home

from rag import (
    ChatState,
    Config,
    QueryState,
    RAGWorkflow,
    clean_special_tokens,
    error,
)
from rag import (
    cmd_chat,
    cmd_index,
    cmd_query,
    cmd_search,
    cmd_serve,
)
# from rag_client.exceptions import ConfigurationError, RAGClientError
# from rag_client.utils.logging import get_logger, setup_logging


# Define a class to hold the parsed arguments
class Args(TypedArgs):
    from_: Optional[str]
    num_workers: Optional[int]
    recursive: bool
    verbose: bool
    debug: bool
    streaming: bool
    host: str
    port: int
    reload_server: bool
    config: str
    top_k: Optional[int]
    sparse_top_k: Optional[int]
    command: str
    args: List[str]


def parse_args(arguments: List[str] = sys.argv[1:]) -> Args:
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
    _ = parser.add_argument("--debug", action="store_true", help="Debug?")
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
    _ = parser.add_argument("--top-k", type=int, help="Top number of chunks to return")
    _ = parser.add_argument(
        "--sparse-top-k", type=int, help="Top number of chunks to return (sparse)"
    )
    _ = parser.add_argument("command")
    _ = parser.add_argument("args", nargs=argparse.REMAINDER)

    _ = parser.parse_known_args(arguments)

    return Args.from_argparse(parser.parse_args())


# CLI commands moved to rag_client.cli.commands
from rag import execute_command, rag_initialize


def main(args: Args) -> None:
    """
    The main entry point.

    Args:
        args: The parsed arguments.
    """
    # Determine log level based on arguments
    if args.debug:
        log_level = "DEBUG"
    elif args.verbose:
        log_level = "INFO"
    else:
        log_level = "WARNING"
    
    # Setup logging using our centralized configuration
    setup_logging(
        level=log_level,
        colored=True,
        logger_configs={
            "rag": log_level,
            "rag_client": log_level,
            "llama_index": "WARNING" if log_level != "DEBUG" else "DEBUG",
        }
    )
    
    # Get logger for this module
    logger = get_logger("rag")

    try:
        # Initialize the RAG workflow and retriever
        logger.info(f"Initializing RAG workflow with config: {args.config}")
        (rag, retriever) = rag_initialize(
            logger=logger,
            config_path=Path(args.config),
            input_from=args.from_,
            num_workers=args.num_workers,
            recursive=args.recursive,
            index_files=args.command == "index",
            top_k=args.top_k,
            sparse_top_k=args.sparse_top_k,
            verbose=args.verbose or args.debug,
        )
        logger.debug("RAG workflow initialized successfully")
        
        # Handle different commands
        logger.info(f"Executing command: {args.command}")
        match args.command:
            case "serve":
                # Import and start the API server
                from rag import cmd_serve

                # Set the API workflow (if api module is used)
                # api.workflow = rag
                logger.info(f"Starting API server on {args.host}:{args.port}")
                # Start the API server
                cmd_serve(
                    host=args.host,
                    port=args.port,
                    reload=args.reload_server,
                )
            case "index":
                # Currently a no-op (indexing done during initialization)
                logger.info("Indexing completed")
                pass
            case _:
                # Execute the command
                execute_command(logger, rag, retriever, args)
                
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except RAGClientError as e:
        logger.error(f"RAG client error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set an environment variable to avoid tokenizer parallelism issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Parse arguments and execute the main function
    main(parse_args())
