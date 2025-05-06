#!/usr/bin/env python

import asyncio
import json
import logging
import os
import sys

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
