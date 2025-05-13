#!/usr/bin/env python
# pyright: reportUnknownVariableType=false

import json
import logging
import sys
import argparse

from typed_argparse import TypedArgs
from xdg_base_dirs import xdg_config_home

from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    StreamingResponse,
    Response,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.storage.chat_store import SimpleChatStore

from rag import *
import api


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
    parser = argparse.ArgumentParser()

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
        "--config", "-c", type=str, help="Yaml config file, to set argument defaults"
    )
    _ = parser.add_argument("command")
    _ = parser.add_argument("args", nargs=argparse.REMAINDER)

    _ = parser.parse_known_args(arguments)

    return Args.from_argparse(parser.parse_args())


def search_command(rag: RAGWorkflow, retriever: BaseRetriever, query: str):
    nodes = rag.retrieve_nodes(retriever, query)
    print(json.dumps(nodes, indent=2))


def query_command(query_state: QueryState, query: str):
    response: RESPONSE_TYPE = query_state.query(query=query)
    match response:
        case StreamingResponse():
            for token in response.response_gen:
                token = clean_special_tokens(token)
                print(token, end="", flush=True)
            print()
        case Response():
            print(response.response)
        case _:
            error(f"query_command cannot render response: {response}")


def chat_command(chat_state: ChatState, query: str, streaming: bool):
    response: StreamingAgentChatResponse | AgentChatResponse = chat_state.chat(
        query=query,
        streaming=streaming,
    )
    if streaming:
        for token in response.response_gen:
            token = clean_special_tokens(token)
            print(token, end="", flush=True)
        print()
    else:
        print(response.response)


def rag_client(
    rag: RAGWorkflow,
    retriever: BaseRetriever | None,
    args: Args,
):
    match args.command:
        case "search":
            if retriever is not None:
                search_command(rag, retriever, args.args[0])
            else:
                error("Search command requires a retriever")
        case "query":
            query_state = rag.initialize_query(
                retriever=retriever,
                # retries=retries,
                # source_retries=source_retries,
                streaming=args.streaming,
                verbose=args.verbose,
            )
            query_command(query_state, args.args[0])
        case "chat":
            user = rag.config.chat.default_user or "user"

            chat_store_json = xdg_config_home() / "rag-client" / "chat_store.json"
            if rag.config.chat.keep_history:
                chat_store = SimpleChatStore.from_persist_path(str(chat_store_json))
            else:
                chat_store = SimpleChatStore()

            query_state: QueryState | None = None
            chat_state: ChatState | None = None

            while True:
                query = input(f"\n{user}> ")
                if query == "exit":
                    if rag.config.chat.keep_history:
                        chat_store.persist(persist_path=str(chat_store_json))
                    break
                elif query.startswith("search "):
                    if retriever is not None:
                        search_command(rag, retriever, query[7:])
                elif query.startswith("query "):
                    if query_state is None:
                        query_state = rag.initialize_query(
                            retriever=retriever,
                            # retries=retries,
                            # source_retries=source_retries,
                            streaming=args.streaming,
                            verbose=args.verbose,
                        )
                    query_command(query_state, query[6:])
                else:
                    if chat_state is None:
                        chat_state = rag.initialize_chat(
                            retriever=retriever,
                            verbose=args.verbose,
                        )
                    chat_command(
                        chat_state,
                        query=query,
                        streaming=args.streaming,
                    )
        case _:
            error(f"Command unrecognized: {args.command}")


def main(args: Args):
    logging.basicConfig(
        stream=sys.stdout,
        encoding="utf-8",
        level=logging.INFO if args.verbose else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",  # Log message format
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger("rag")

    (rag, retriever) = rag_initialize(
        logger=logger,
        config_path=Path(args.config),
        input_from=args.from_,
        num_workers=args.num_workers,
        recursive=args.recursive,
        verbose=args.verbose,
    )

    match args.command:
        case "serve":
            api.workflow = rag
            api.start_api_server(
                host=args.host,
                port=args.port,
                reload=args.reload_server,
            )
        case "index":
            pass
        case _:
            rag_client(rag, retriever, args)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(parse_args())
