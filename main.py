#!/usr/bin/env python
# pyright: reportUnknownVariableType=false

import asyncio
import json
import logging
import os
import sys
import yaml
import argparse

from xdg_base_dirs import xdg_config_home

from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    AsyncStreamingResponse,
    Response,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.storage.chat_store import SimpleChatStore

from rag import *
import api


async def search_command(rag: RAGWorkflow, query: str):
    nodes = await rag.retrieve_nodes(query)
    print(json.dumps(nodes, indent=2))


async def query_command(
    rag: RAGWorkflow,
    query: str,
    streaming: bool,
    retries: bool = False,
    source_retries: bool = False,
):
    response: RESPONSE_TYPE = await rag.query(
        query,
        retries=retries,
        source_retries=source_retries,
        streaming=streaming,
    )
    match response:
        case AsyncStreamingResponse():
            async for token in response.async_response_gen():
                token = clean_special_tokens(token)
                print(token, end="", flush=True)
            print()
        case Response():
            print(response.response)
        case _:
            error(f"query_command cannot render response: {response}")


async def chat_command(
    rag: RAGWorkflow,
    user: str,
    query: str,
    streaming: bool,
    token_limit: int,
    chat_store: SimpleChatStore | None = None,
    summarize_chat: bool = False,
):
    response: StreamingAgentChatResponse | AgentChatResponse = await rag.chat(
        user=user,
        query=query,
        token_limit=token_limit,
        chat_store=chat_store,
        streaming=streaming,
        summarize_chat=summarize_chat,
    )
    if streaming:
        async for token in response.async_response_gen():
            token = clean_special_tokens(token)
            print(token, end="", flush=True)
        print()
    else:
        print(response.response)


async def rag_client(rag: RAGWorkflow, args: Args):
    match args.command:
        case "search":
            await search_command(rag, args.args[0])
        case "query":
            await query_command(
                rag,
                args.args[0],
                streaming=args.streaming,
                retries=args.retries,
                source_retries=args.source_retries,
            )
        case "chat":
            user = args.chat_user or "user"

            chat_store_json = xdg_config_home() / "rag-client" / "chat_store.json"
            if args.chat_user is not None:
                chat_store = SimpleChatStore.from_persist_path(str(chat_store_json))
            else:
                chat_store = SimpleChatStore()

            while True:
                query = input(f"\n{user}> ")
                if query == "exit":
                    if args.chat_user is not None:
                        chat_store.persist(persist_path=str(chat_store_json))
                    break
                elif query.startswith("search "):
                    await search_command(rag, query[7:])
                elif query.startswith("query "):
                    await query_command(
                        rag,
                        query[6:],
                        streaming=args.streaming,
                        retries=args.retries,
                        source_retries=args.source_retries,
                    )
                else:
                    await chat_command(
                        rag,
                        user=user,
                        query=query,
                        token_limit=args.token_limit,
                        chat_store=chat_store,
                        streaming=args.streaming,
                    )
        case _:
            error(f"Command unrecognized: {args.command}")


def parse_args(
    arguments: list[str] = sys.argv[1:], config_path: Path | None = None
) -> Args:
    parser = argparse.ArgumentParser()

    _ = parser.add_argument(
        "--config", "-c", type=str, help="Yaml config file, to set argument defaults"
    )
    _ = parser.add_argument("--verbose", action="store_true", help="Verbose?")
    _ = parser.add_argument(
        "--db-conn",
        type=str,
        help="Postgres connection string (in-memory if unspecified)",
    )
    _ = parser.add_argument(
        "--hnsw-m",
        type=int,
        default=16,
        help="Bi-dir links for each node (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=64,
        help="Dynamic candidate list size (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=40,
        help="Candidate list size during search (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--hnsw-dist-method",
        type=str,
        default="vector_cosine_ops",
        help="Distance method for similarity (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--embed-provider", type=str, help="Embedding model provider"
    )
    _ = parser.add_argument("--embed-model", type=str, help="Embedding model")
    _ = parser.add_argument("--embed-api-key", type=str, help="Embedding model API key")
    _ = parser.add_argument(
        "--embed-api-version", type=str, help="Embedding model API version"
    )
    _ = parser.add_argument(
        "--embed-base-url",
        type=str,
        help="Embedding model base URL",
    )
    _ = parser.add_argument(
        "--embed-dim",
        type=int,
        default=512,
        help="Embedding dimensions (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--query-instruction", type=str, help="Query instruction for embedding model"
    )
    _ = parser.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size (default: %(default)s)"
    )
    _ = parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=20,
        help="Chunk overlap (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--splitter",
        type=str,
        default="Sentence",
        help="Document splitting strategy (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--buffer-size",
        type=int,
        default=256,
        help="Buffer size for semantic splitting (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--breakpoint-percentile-threshold",
        type=int,
        default=95,
        help="Breakpoint percentile threshold (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Window size of sentence window splitter (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--questions-answered",
        type=int,
        help="If provided, generate N questions related to each chunk",
    )
    _ = parser.add_argument(
        "--questions-answered-provider",
        type=str,
        help="Questions answered model provider",
    )
    _ = parser.add_argument(
        "--questions-answered-model", type=str, help="Questions answered model"
    )
    _ = parser.add_argument(
        "--questions-answered-api-key",
        type=str,
        help="Questions answered model API key",
    )
    _ = parser.add_argument(
        "--questions-answered-api-version",
        type=str,
        help="Questions answered model API version",
    )
    _ = parser.add_argument(
        "--questions-answered-base-url",
        type=str,
        help="Questions answered model base URL",
    )
    _ = parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top K document nodes (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--llm-provider",
        type=str,
        help="Provider for LLM used for text generation and chat",
    )
    _ = parser.add_argument(
        "--llm-model",
        type=str,
        help="LLM used for text generation and chat",
    )
    _ = parser.add_argument(
        "--llm-api-key",
        type=str,
        default="fake",
        help="API key to use with LLM",
    )
    _ = parser.add_argument(
        "--llm-api-version",
        type=str,
        help="API version to use with LLM (if required)",
    )
    _ = parser.add_argument(
        "--llm-base-url",
        type=str,
        help="URL to use for talking with LLM (default depends on LLM)",
    )
    _ = parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream output as it arrives from LLM",
    )
    _ = parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Max time to wait in seconds (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="LLM temperature value (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="LLM maximum answer size in tokens (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--context-window",
        type=int,
        default=8192,
        help="LLM context window size (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="LLM reasoning effort (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Number of GPU layers to use (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--chat-user",
        type=str,
        help="Chat user name for history saves (no history if unset)",
    )
    _ = parser.add_argument(
        "--token-limit",
        type=int,
        default=1500,
        help="Token limit used for chat history (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--from", dest="from_", type=str, help="Where to read files from (optional)"
    )
    _ = parser.add_argument(
        "--recursive",
        action="store_true",
        help="Read directories recursively (default: no)",
    )
    _ = parser.add_argument(
        "--metadata-extractor-provider",
        type=str,
        help="Metadata extractor model provider",
    )
    _ = parser.add_argument(
        "--metadata-extractor-model", type=str, help="Metadata extractor model"
    )
    _ = parser.add_argument(
        "--metadata-extractor-api-key",
        type=str,
        help="Metadata extractor model API key",
    )
    _ = parser.add_argument(
        "--metadata-extractor-api-version",
        type=str,
        help="Metadata extractor model API version",
    )
    _ = parser.add_argument(
        "--metadata-extractor-base-url",
        type=str,
        help="Metadata extractor model base URL",
    )
    _ = parser.add_argument(
        "--collect-keywords",
        action="store_true",
        help="Generate keywords for document retrieval",
    )
    _ = parser.add_argument(
        "--keywords-provider",
        type=str,
        help="Keywords model provider",
    )
    _ = parser.add_argument("--keywords-model", type=str, help="Keywords model")
    _ = parser.add_argument(
        "--keywords-api-key",
        type=str,
        help="Keywords model API key",
    )
    _ = parser.add_argument(
        "--keywords-api-version",
        type=str,
        help="Keywords model API version",
    )
    _ = parser.add_argument(
        "--keywords-base-url",
        type=str,
        help="Keywords model base URL",
    )
    _ = parser.add_argument(
        "--retries",
        action="store_true",
        help="Retry queries based on relevancy",
    )
    _ = parser.add_argument(
        "--source-retries",
        action="store_true",
        help="Retry queries (using source modification) based on relevancy",
    )
    _ = parser.add_argument(
        "--evaluator-provider",
        type=str,
        help="Evaluator model provider",
    )
    _ = parser.add_argument("--evaluator-model", type=str, help="Evaluator model")
    _ = parser.add_argument(
        "--evaluator-api-key",
        type=str,
        help="Evaluator model API key",
    )
    _ = parser.add_argument(
        "--evaluator-api-version",
        type=str,
        help="Evaluator model API version",
    )
    _ = parser.add_argument(
        "--evaluator-base-url",
        type=str,
        help="Evaluator model base URL",
    )
    _ = parser.add_argument(
        "--summarize-chat",
        action="store_true",
        help="Summarize chat history when it grows too long",
    )
    _ = parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of works to use for various tasks (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help='Host to serve from with "serve" command (default: %(default)s)',
    )
    _ = parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help='Port to serve from with "serve" command (default: %(default)s)',
    )
    _ = parser.add_argument(
        "--reload-server",
        action="store_true",
        help='Auto-reload source when using "serve" command (for devel)',
    )
    _ = parser.add_argument("command")
    _ = parser.add_argument("args", nargs=argparse.REMAINDER)

    args: argparse.Namespace
    args, _remaining = parser.parse_known_args(arguments)
    if args.config or config_path is not None:  # pyright: ignore[reportAny]
        with open(
            args.config or str(config_path), "r"  # pyright: ignore[reportAny]
        ) as f:
            config = yaml.safe_load(f)  # pyright: ignore[reportAny]
        parser.set_defaults(**config)

    return Args.from_argparse(parser.parse_args())


def main(args: Args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.basicConfig(
        stream=sys.stdout,
        encoding="utf-8",
        level=logging.INFO if args.verbose else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",  # Log message format
        datefmt="%H:%M:%S",
    )

    rag = asyncio.run(rag_initialize(args))

    match args.command:
        case "serve":
            api.workflow = rag
            api.llm_model = args.llm
            api.embed_model = args.embed_model
            api.token_limit = args.token_limit
            # This cannot run inside asyncio.run, since it creates its own
            # async event loop.
            api.start_api_server(
                host=args.host,
                port=args.port,
                reload=args.reload_server,
            )
        case "index":
            pass
        case _:
            asyncio.run(rag_client(rag, args))


if __name__ == "__main__":
    main(parse_args())
