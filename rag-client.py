#!/usr/bin/env python

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
import pprint

from copy import deepcopy
from functools import cache
from typing import List
from xdg_base_dirs import xdg_cache_home

from llama_index.core import (
    Document,
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    # get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.llms import ChatMessage
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.indices import load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter

# from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.readers.base import BaseReader

# from llama_index.core.response_synthesizers import BaseSynthesizer
# from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import Document

# from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.storage_context import StorageContext

# from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.workflow import (
    # Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from typing import Optional, List, Dict, Union

# from typing import Any
import uuid


### Utility functions


def collection_hash(file_list):
    # List to hold the hash of each file
    file_hashes = []
    for file_path in file_list:
        # Compute SHA-512 hash of the file contents
        h = hashlib.sha512()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        file_hashes.append(h.hexdigest())
    # Concatenate all hashes with newline separators
    concatenated = "\n".join(file_hashes).encode("utf-8")
    # Compute SHA-512 hash of the concatenated hashes
    final_hash = hashlib.sha512(concatenated).hexdigest()
    return final_hash


### Readers


@cache
def get_text_from_org_node(current_node, format: str = "plain") -> List[str]:
    """Extract text from org node. Skip properties"""
    lines = []
    if current_node.heading:
        lines.append(current_node.get_heading(format=format))
    if current_node.body:
        lines.extend(current_node.get_body(format=format).split("\n"))
    for child in current_node.children:
        lines.extend(get_text_from_org_node(child, format=format))
    return lines


class OrgReader(BaseReader):
    """OrgReader

    Extract text from org files.
    Add the :PROPERTIES: on text node as extra_info
    """

    split_depth: int = 0
    text_formatting: str = "plain"  # plain or raw, as supported by orgparse

    def node_to_document(self, node, extra_info):
        """Convert org node to document."""
        text = "\n".join(get_text_from_org_node(node, format=self.text_formatting))
        extra_info = deepcopy(extra_info or {})
        for prop, value in node.properties.items():
            extra_info["org_property_" + prop] = value
        return Document(text=text, extra_info=extra_info)

    def load_data(self, file, extra_info):
        """Parse file into different documents based on root depth."""
        from orgparse import load

        org_content = load(file)
        documents: List[Document] = []

        extra_info = extra_info or {}
        extra_info["filename"] = org_content.env.filename

        # In orgparse, list(org_content) ALL the nodes in the file
        # So we use this to process the nodes below the split_depth as
        # separate documents and skip the rest. This means at a split_depth
        # of 2, we make documents from nodes at levels 0 (whole file), 1, and 2.
        # The text will be present in multiple documents!
        for node in list(org_content):
            if node.level <= self.split_depth:
                documents.append(self.node_to_document(node, extra_info))

        return documents


### Ingestion


def load_vector_store(args):
    if args.db_name is not None:
        if args.verbose:
            print("Setup PostgreSQL vector store")
        vector_store = PGVectorStore.from_params(
            database=args.db_name,
            host=args.db_host,
            password=args.db_pass or "",
            port=str(args.db_port),
            user=args.db_user,
            table_name=args.db_table,
            embed_dim=args.embed_dim,
            hnsw_kwargs={
                "hnsw_m": args.hnsw_m,
                "hnsw_ef_construction": args.hnsw_ef_construction,
                "hnsw_ef_search": args.hnsw_ef_search,
                "hnsw_dist_method": args.hnsw_dist_method,
            },
            # create_engine_kwargs={
            #     "connect_args": {"options": "-c client_encoding=UTF8"}
            # },
        )
    else:
        if args.verbose:
            print("Use in-memory vector store")
        vector_store = None
    return vector_store


def load_vector_index(
    input_files,
    embed_model,
    vector_store=None,
    chunk_size=None,
    chunk_overlap=None,
    verbose=False,
):
    # Determine the cache directory as a unique function of the inputs
    cache_dir = xdg_cache_home() / "rag-client"
    cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    if input_files is not None and vector_store is None:
        fingerprint = [
            collection_hash(input_files),
            hashlib.sha512(embed_model.encode("utf-8")).hexdigest(),
            hashlib.sha512(str(chunk_size).encode("utf-8")).hexdigest(),
            hashlib.sha512(str(chunk_overlap).encode("utf-8")).hexdigest(),
        ]
        final_hash = "\n".join(fingerprint).encode("utf-8")
        final_base64 = base64.b64encode(final_hash).decode("utf-8")[0:32]

        persist_dir = cache_dir / final_base64
        if verbose:
            print(f"Cache directory = {persist_dir}")
    else:
        persist_dir = None

    if verbose:
        print("Load HuggingFace embedding model")
    # jww (2025-05-01): Need to support multiple embedding classes here,
    # probably by parsing more complex embed_model strings, such as:
    # HuggingFace/MODEL
    embed_model = HuggingFaceEmbedding(model_name=embed_model)

    if input_files is None:
        if vector_store is not None:
            if verbose:
                print("Load index from database")
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model,
                show_progress=verbose,
            )
        else:
            print("No vector store and no input files to index!", file=sys.stderr)
            sys.exit(1)

    elif persist_dir is not None and os.path.exists(persist_dir):
        # If a cache dir was specified and exists, load the index
        global Settings
        Settings.embed_model = embed_model
        if chunk_size is not None:
            Settings.chunk_size = chunk_size
        if chunk_overlap is not None:
            Settings.chunk_overlap = chunk_overlap

        if verbose:
            print("Load index from cache")
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(persist_dir))
        )
    else:
        file_extractor = {".org": OrgReader()}

        if verbose:
            print("Read documents from disk")
        documents = SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=file_extractor,  # type: ignore[override]
            recursive=True,
        ).load_data()

        if chunk_size is not None and chunk_overlap is not None:
            if verbose:
                print("Chunk documents into sentences")
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            nodes = splitter.get_nodes_from_documents(documents)
        else:
            if verbose:
                print("Use whole documents")
            nodes = SimpleNodeParser().get_nodes_from_documents(documents)

        if vector_store is not None:
            if verbose:
                print("Create storage context")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
        else:
            storage_context = None

        if verbose:
            print("Calculate vector embeddings")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=verbose,
        )

        # If a cache dir was specified, persist the index
        if persist_dir is not None:
            if vector_store is not None:
                if verbose:
                    print("Note index is in database")
                persist_dir.touch()
            else:
                if verbose:
                    print("Write index to cache")
                index.storage_context.persist(persist_dir=persist_dir)

    return index


### Query workflow


class VectorSearchEvent(Event):
    query: str
    mode: str
    chat_id: Optional[str] = None


class VectorSearchResultEvent(Event):
    nodes: List[Dict]
    query: str
    mode: str
    chat_id: Optional[str] = None


class LLMQueryEvent(Event):
    query: str
    context_nodes: List[Dict]
    mode: str
    chat_history: Optional[List[ChatMessage]] = None
    chat_id: Optional[str] = None


class LLMResponseEvent(Event):
    response: str
    query: str
    mode: str
    chat_history: Optional[List[ChatMessage]] = None
    chat_id: Optional[str] = None


class RAGWorkflow(Workflow):
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        similarity_top_k: int = 3,
        llm: BaseLLM | None = None,
        memory: Optional[ChatMemoryBuffer] = None,
        token_limit: int = 1500,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vector_index = vector_index
        self.similarity_top_k = similarity_top_k
        self.llm = llm
        self.memory = memory or ChatMemoryBuffer.from_defaults(token_limit=token_limit)
        self.verbose = verbose

    @step
    async def start(self, event: StartEvent) -> VectorSearchEvent:
        """Initialize workflow based on mode selection"""
        mode = event.mode
        chat_id = event.chat_id or str(uuid.uuid4()) if mode == "chat" else None
        if mode == "chat":
            self.memory.put(ChatMessage(role="user", content=event.query))
        return VectorSearchEvent(query=event.query, mode=mode, chat_id=chat_id)

    @step
    async def vector_search(self, event: VectorSearchEvent) -> VectorSearchResultEvent:
        """Perform vector similarity search"""
        if self.verbose:
            print("Create retriever object")
        retriever = self.vector_index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )
        if self.verbose:
            print("Retrieve nodes from vector index")
        nodes = await retriever.aretrieve(event.query)
        if self.verbose:
            print(f"{len(nodes)} nodes found in vector index")
        return VectorSearchResultEvent(
            nodes=[{"text": node.text, "metadata": node.metadata} for node in nodes],
            query=event.query,
            mode=event.mode,
            chat_id=event.chat_id,
        )

    @step
    async def process_search_results(
        self, event: VectorSearchResultEvent
    ) -> Union[StopEvent, LLMQueryEvent]:
        """Route based on workflow mode"""
        if event.mode == "vector_search":
            return StopEvent(
                result={
                    "query": event.query,
                    "results": event.nodes,
                    "mode": event.mode,
                }
            )

        return LLMQueryEvent(
            query=event.query,
            context_nodes=event.nodes,
            mode=event.mode,
            chat_history=(
                self.memory.get(event.chat_id) if event.mode == "chat" else None
            ),
            chat_id=event.chat_id,
        )

    @step
    async def llm_query(self, event: LLMQueryEvent) -> LLMResponseEvent:
        """Execute LLM query with context"""
        context_str = "\n".join([n["text"] for n in event.context_nodes])
        chat_history_str = (
            "\n".join([f"{msg.role}: {msg.content}" for msg in event.chat_history])
            if event.chat_history
            else ""
        )

        if self.verbose:
            print("Building LLM prompt")
        qa_prompt = PromptTemplate(
            "CONTEXT information is below:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "CHAT HISTORY: {chat_history_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "QUERY: {query_str}\n"
            "ANSWER: "
        )

        if self.llm is not None:
            if self.verbose:
                print("Building query string")
            query = qa_prompt.format(
                context_str=context_str,
                chat_history_str=chat_history_str,
                query_str=event.query,
            )
            if self.verbose:
                print("Submitting query to LLM")
            response = await self.llm.acomplete(query)
            response = response.text
        else:
            response = ""
        return LLMResponseEvent(
            response=response,
            query=event.query,
            mode=event.mode,
            chat_id=event.chat_id,
        )

    @step
    async def finalize_response(self, event: LLMResponseEvent) -> StopEvent:
        """Handle final response formatting"""
        if event.mode == "chat":
            if self.verbose:
                print("Adding chat response to memory")
            self.memory.put(ChatMessage(role="assistant", content=event.response))
            return StopEvent(
                result={
                    "response": event.response,
                    "chat_id": event.chat_id,
                    "chat_history": self.memory.get(event.chat_id),
                    "mode": event.mode,
                }
            )
        else:
            return StopEvent(result={"response": event.response, "mode": event.mode})


### Command functions


async def exec_search(workflow, query):
    if workflow.verbose:
        print("Execute vector search only")
    vector_result = await workflow.run(query=query, mode="vector_search")
    print(json.dumps(vector_result["results"], indent=2))


async def exec_query(workflow, query):
    if workflow.verbose:
        print("Execute vector search + LLM query")
    query_result = await workflow.run(query=query, mode="llm_query")
    print(query_result["response"])


async def exec_chat(workflow):
    if workflow.verbose:
        print("Execute persistant, interactive chat")
    chat_result = None
    while True:
        query = input("> ")
        if query.lower() == "exit":
            break
        elif query.lower() == "history":
            if chat_result is not None:
                pprint.pprint(chat_result["chat_history"])
            else:
                print("There is no history yet")
        else:
            if chat_result is not None:
                chat_id = chat_result["chat_id"]
            else:
                chat_id = None

            chat_result = await workflow.run(query=query, mode="chat", chat_id=chat_id)
            print(chat_result["response"])


### Main function


def main():
    parser = argparse.ArgumentParser(description="Read long command-line options")

    parser.add_argument(
        "--db-name",
        type=str,
        help="Postgres db (in-memory vector index if unspecified)",
    )
    parser.add_argument(
        "--db-host",
        type=str,
        default="localhost",
        help="Postgres db host (default: %(default)s)",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=5432,
        help="Postgres db port (default: %(default)s)",
    )
    parser.add_argument(
        "--db-user",
        type=str,
        default="postgres",
        help="Postgres db user (default: %(default)s)",
    )
    parser.add_argument("--db-pass", type=str, help="Postgres db password")
    parser.add_argument(
        "--db-table",
        type=str,
        default="vectors",
        help="Postgres db table (default: %(default)s)",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=16,
        help="Bi-dir links for each node (default: %(default)s)",
    )
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=64,
        help="Dynamic candidate list size (default: %(default)s)",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=40,
        help="Candidate list size during search (default: %(default)s)",
    )
    parser.add_argument(
        "--hnsw-dist-method",
        type=str,
        default="vector_cosine_ops",
        help="Distance method for similarity (default: %(default)s)",
    )
    parser.add_argument("--embed-model", type=str, help="Embedding model")
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=512,
        help="Embedding dimensions (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size (default: %(default)s)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=20,
        help="Chunk overlap (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top K document nodes (default: %(default)s)",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=1500,
        help="Token limit for chat history (not --search or --query)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose?")
    parser.add_argument("--read-files", action="store_true", help="Read files?")
    parser.add_argument(
        "--search", type=str, help="Search documents for closely text matching query"
    )
    parser.add_argument(
        "--query", type=str, help="Submit query to LLM, with matching text in context"
    )

    args = parser.parse_args()

    if args.read_files:
        input_files = [line.strip() for line in sys.stdin if line.strip()]
        if not input_files:
            print("No filenames provided on standard input", file=sys.stderr)
            sys.exit(1)
    else:
        input_files = None

    if args.verbose:
        print("Load vector store")
    vector_store = load_vector_store(args)

    if args.verbose:
        print("Load vector index")
    index = load_vector_index(
        input_files=input_files,
        embed_model=args.embed_model,
        vector_store=vector_store,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        verbose=args.verbose,
    )

    if args.search is None:
        # jww (2025-05-01): Need to support multiple LLM classes here,
        # probably by parsing more complex embed_model strings, such as:
        # LlamaCpp/MODEL
        if args.verbose:
            print("Create LLM object")
        llm = LlamaCPP(
            # model_url=model_url,
            model_path="/Users/johnw/Models/Qwen_Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q8_0.gguf",
            temperature=0.6,
            max_new_tokens=256,
            context_window=16384,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": -1},
            verbose=args.verbose,
        )
    else:
        llm = None

    if args.verbose:
        print("Create RAG workflow")
    workflow = RAGWorkflow(
        vector_index=index,
        similarity_top_k=args.top_k,
        llm=llm,
        token_limit=args.token_limit,
        verbose=args.verbose,
    )

    if args.search is not None:
        asyncio.run(exec_search(workflow, args.search))
    elif args.query is not None:
        asyncio.run(exec_query(workflow, args.query))
    else:
        asyncio.run(exec_chat(workflow))


if __name__ == "__main__":
    main()
