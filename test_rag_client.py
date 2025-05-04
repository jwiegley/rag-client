#!/usr/bin/env python

import asyncio

from rag_client import *


def test_foo():
    args = Args(
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
        db_name=None,
        ####################
        command="",
        args=[],
        from_="/Users/johnw/org/conference/202410151104-ethdenver-denver-2025.org",
        recursive=False,
    )

    async def query_documents() -> list[str]:
        rag_workflow = RAGWorkflow(args)
        await rag_workflow.initialize()
        nodes = await rag_workflow.retrieve_nodes("What did Hafsah say?")
        return [node["text"] for node in nodes]

    result = asyncio.run(query_documents())
    assert len(result) > 1
