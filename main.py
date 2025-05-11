#!/usr/bin/env python
# pyright: reportUnknownVariableType=false

import asyncio
import json
import logging
import sys
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


async def search_command(rag: RAGWorkflow, retriever: BaseRetriever, query: str):
    nodes = await rag.retrieve_nodes(retriever, query)
    print(json.dumps(nodes, indent=2))


async def query_command(
    rag: RAGWorkflow,
    models: Models,
    retriever: BaseRetriever | None,
    query: str,
    streaming: bool,
    retries: bool = False,
    source_retries: bool = False,
):
    response: RESPONSE_TYPE = await rag.query(
        models,
        retriever,
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
    models: Models,
    retriever: BaseRetriever | None,
    user: str,
    query: str,
    streaming: bool,
    token_limit: int,
    chat_store: SimpleChatStore | None = None,
    summarize_chat: bool = False,
):
    response: StreamingAgentChatResponse | AgentChatResponse = await rag.chat(
        models,
        retriever,
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


async def rag_client(
    rag: RAGWorkflow,
    models: Models,
    retriever: BaseRetriever | None,
    args: Args,
):
    match args.command:
        case "search":
            if retriever is not None:
                await search_command(rag, retriever, args.args[0])
            else:
                error("Search command requires a retriever")
        case "query":
            await query_command(
                rag,
                models,
                retriever,
                args.args[0],
                streaming=rag.config.streaming,
                retries=rag.config.retries,
                source_retries=rag.config.source_retries,
            )
        case "chat":
            user = rag.config.chat_user or "user"

            chat_store_json = xdg_config_home() / "rag-client" / "chat_store.json"
            if rag.config.chat_user is not None:
                chat_store = SimpleChatStore.from_persist_path(str(chat_store_json))
            else:
                chat_store = SimpleChatStore()

            while True:
                query = input(f"\n{user}> ")
                if query == "exit":
                    if rag.config.chat_user is not None:
                        chat_store.persist(persist_path=str(chat_store_json))
                    break
                elif query.startswith("search "):
                    if retriever is not None:
                        await search_command(rag, retriever, query[7:])
                elif query.startswith("query "):
                    await query_command(
                        rag,
                        models,
                        retriever,
                        query[6:],
                        streaming=rag.config.streaming,
                        retries=rag.config.retries,
                        source_retries=rag.config.source_retries,
                    )
                else:
                    await chat_command(
                        rag,
                        models,
                        retriever,
                        user=user,
                        query=query,
                        token_limit=rag.config.token_limit,
                        chat_store=chat_store,
                        streaming=rag.config.streaming,
                    )
        case _:
            error(f"Command unrecognized: {args.command}")


def parse_args(arguments: list[str] = sys.argv[1:]) -> Args:
    parser = argparse.ArgumentParser()

    _ = parser.add_argument(
        "--from", dest="from_", type=str, help="Where to read files from (optional)"
    )
    _ = parser.add_argument("--verbose", action="store_true", help="Verbose?")
    _ = parser.add_argument(
        "--config", "-c", type=str, help="Yaml config file, to set argument defaults"
    )
    _ = parser.add_argument("command")
    _ = parser.add_argument("args", nargs=argparse.REMAINDER)

    _ = parser.parse_known_args(arguments)

    return Args.from_argparse(parser.parse_args())


def main(args: Args):
    logging.basicConfig(
        stream=sys.stdout,
        encoding="utf-8",
        level=logging.INFO if args.verbose else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",  # Log message format
        datefmt="%H:%M:%S",
    )

    (rag, models, retriever) = asyncio.run(rag_initialize(args))

    match args.command:
        case "serve":
            api.workflow = rag
            api.llm_model = rag.config.llm_model
            api.embedding = rag.config.embedding
            api.token_limit = rag.config.token_limit
            # This cannot run inside asyncio.run, since it creates its own
            # async event loop.
            api.start_api_server(
                host=rag.config.host,
                port=rag.config.port,
                reload=rag.config.reload_server,
            )
        case "index":
            pass
        case _:
            asyncio.run(rag_client(rag, models, retriever, args))


if __name__ == "__main__":
    main(parse_args())
