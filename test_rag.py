#!/usr/bin/env python

import asyncio

from rag import *


def test_foo():
    args = Args(
        config=None,
        db_conn=None,
        hnsw_m=0,
        hnsw_ef_construction=0,
        hnsw_ef_search=0,
        hnsw_dist_method="",
        embed_provider=None,
        splitter="Sentence",
        buffer_size=0,
        breakpoint_percentile_threshold=0,
        window_size=0,
        llm_provider=None,
        llm_api_key="",
        temperature=1.0,
        max_tokens=1500,
        context_window=8192,
        reasoning_effort="medium",
        gpu_layers=-1,
        chat_user="user1",
        token_limit=1500,
        collect_keywords=False,
        retries=False,
        source_retries=False,
        summarize_chat=False,
        num_workers=4,
        host="localhost",
        port=8000,
        reload_server=False,
        verbose=True,
        embed_model="HuggingFace:BAAI/bge-large-en-v1.5",
        embed_base_url=None,
        embed_dim=1024,
        chunk_size=512,
        chunk_overlap=20,
        top_k=20,
        timeout=600,
        llm="OpenAILike:Falcon3-10B-Instruct",
        llm_base_url="http://192.168.50.5:8080/v1",
        llm_api_version=None,
        streaming=True,
        questions_answered=None,
        ####################
        command="",
        args=[],
        from_="/Users/johnw/org/conference/202410151104-ethdenver-denver-2025.org",
        recursive=False,
    )

    async def query_documents() -> list[str]:
        rag = RAGWorkflow(verbose=args.verbose)
        await rag.initialize(args)
        nodes = await rag.retrieve_nodes("What did Hafsah say?")
        return [node["text"] for node in nodes]

    result = asyncio.run(query_documents())
    assert len(result) > 1
