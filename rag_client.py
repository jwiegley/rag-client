#!/usr/bin/env python

import asyncio
import base64
import hashlib
import json
import logging
import os
import sys
import yaml
import argparse
import psycopg2

from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    AsyncStreamingResponse,
    Response,
)
from llama_index.core.response_synthesizers import ResponseMode

from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from orgparse.node import OrgNode
from pathlib import Path
from typed_argparse import TypedArgs
from typing import Any, Literal, NoReturn, cast, final, no_type_check, override
from xdg_base_dirs import xdg_cache_home, xdg_config_home

from llama_index.core import (
    KeywordTableIndex,
    QueryBundle,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
    VectorStoreIndex,
    load_indices_from_storage,  # pyright: ignore[reportUnknownVariableType]
)
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from llama_index.core.storage.storage_context import DEFAULT_PERSIST_DIR
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import (
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.keyword_table.base import BaseKeywordTableIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
)
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    RetryQueryEngine,
    RetrySourceQueryEngine,
)
from llama_index.core.evaluation import GuidelineEvaluator, RelevancyEvaluator
from llama_index.core.readers.base import BaseReader
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    BaseNode,
    Document,
    Node,
    NodeWithScore,
    TransformComponent,
)
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.vector_stores.simple import SimpleVectorStore

from llama_index.embeddings.huggingface import (  # pyright: ignore[reportMissingTypeStubs]
    HuggingFaceEmbedding,
)
from llama_index.embeddings.ollama import (  # pyright: ignore[reportMissingTypeStubs]
    OllamaEmbedding,
)
from llama_index.embeddings.openai import (  # pyright: ignore[reportMissingTypeStubs]
    OpenAIEmbedding,
)
from llama_index.embeddings.openai_like import (  # pyright: ignore[reportMissingTypeStubs]
    OpenAILikeEmbedding,
)
from llama_index.llms.llama_cpp import (  # pyright: ignore[reportMissingTypeStubs]
    LlamaCPP,
)
from llama_index.llms.ollama import Ollama  # pyright: ignore[reportMissingTypeStubs]
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import (  # pyright: ignore[reportMissingTypeStubs]
    OpenAILike,
)
from llama_index.vector_stores.postgres import (  # pyright: ignore[reportMissingTypeStubs]
    PGVectorStore,
)
from llama_index.storage.docstore.postgres import (  # pyright: ignore[reportMissingTypeStubs]
    PostgresDocumentStore,
)
from llama_index.storage.index_store.postgres import (  # pyright: ignore[reportMissingTypeStubs]
    PostgresIndexStore,
)

IndexList = list[BaseIndex[IndexDict]]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

### Utility functions


logger = logging.getLogger("rag_client")


async def awaitable_none():
    return None


def error(msg: str) -> NoReturn:
    print(msg, sys.stderr)
    sys.exit(1)


def parse_prefixes(prefixes: list[str], s: str) -> tuple[str | None, str]:
    for prefix in prefixes:
        if s.startswith(prefix):
            return prefix, s[len(prefix) :]
    return None, s  # No matching prefix found


def list_files(directory: Path, recursive: bool = False) -> list[Path]:
    if recursive:
        file_list: list[Path] = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file not in [".", ".."]:
                    file_list.append(Path(root) / Path(file))
        return file_list
    else:
        return [
            directory / f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]


def read_files(read_from: str, recursive: bool = False) -> list[Path] | NoReturn:
    if read_from == "-":
        input_files = [Path(line.strip()) for line in sys.stdin if line.strip()]
        if not input_files:
            error("No filenames provided on standard input")
        return input_files
    elif os.path.isfile(read_from):
        return [Path(read_from)]
    elif os.path.isdir(read_from):
        return list_files(Path(read_from), recursive)
    else:
        error(f"Input path is unrecognized or non-existent: {read_from}")


def collection_hash(file_list: list[Path]) -> str:
    # List to hold the hash of each file
    file_hashes: list[str] = []
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


def cache_dir(fingerprint: str) -> Path:
    cache_dir = xdg_cache_home() / "rag-client"
    cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    return cache_dir / fingerprint


def clean_special_tokens(text: str) -> str:
    # Remove <|assistant|> with various newline combinations
    patterns = [
        "<|assistant|>\n\n",
        "<|assistant|>\n",
        "\n\n<|assistant|>",
        "\n<|assistant|>",
        "<|assistant|>",
    ]
    for pattern in patterns:
        text = text.replace(pattern, "")
    return text


### Args


class Args(TypedArgs):
    config: str | None
    verbose: bool
    db_conn: str | None
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    hnsw_dist_method: str
    embed_model: str | None
    embed_base_url: str | None
    embed_dim: int
    chunk_size: int
    chunk_overlap: int
    splitter: str
    buffer_size: int
    breakpoint_percentile_threshold: int
    window_size: int
    questions_answered: int | None
    top_k: int
    llm: str | None
    llm_api_key: str
    llm_api_version: str | None
    llm_base_url: str | None
    streaming: bool
    timeout: int
    temperature: float
    max_tokens: int
    context_window: int
    reasoning_effort: Literal["low", "medium", "high"]
    gpu_layers: int
    chat_user: str | None
    token_limit: int
    from_: str | None
    recursive: bool
    collect_keywords: bool
    retries: bool
    source_retries: bool
    summarize_chat: bool
    num_workers: int
    command: str
    args: list[str]


### Readers


@cache
@no_type_check
def get_text_from_org_node(current_node: OrgNode, format: str = "plain"):
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

    @no_type_check
    def node_to_document(self, node: Node, extra_info):
        """Convert org node to document."""
        text = "\n".join(get_text_from_org_node(node, format=self.text_formatting))
        extra_info = deepcopy(extra_info or {})
        for prop, value in node.properties.items():
            extra_info["org_property_" + prop] = value
        return Document(text=text, extra_info=extra_info)

    @no_type_check
    def load_data(self, file, extra_info):
        """Parse file into different documents based on root depth."""
        from orgparse import load

        org_content = load(file)
        documents: list[Document] = []

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


### Embeddings


@no_type_check
def flatten_floats(data: list[float] | list[list[float]]) -> list[float]:
    if not data:
        return []
    # Check if the first element is a float (covers the List[float] case)
    if isinstance(data[0], float):
        return data  # type: ignore
    # Otherwise, assume it's List[List[float]]
    return [item for sublist in data for item in sublist]


class LlamaCppEmbedding(BaseEmbedding):
    @no_type_check
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        from llama_cpp import Llama

        self._model = Llama(model_path=model_path, embedding=True)

    @override
    def _get_text_embedding(self, text: str) -> Embedding:
        response = self._model.create_embedding(text)
        return flatten_floats(response["data"][0]["embedding"])

    @override
    def _get_query_embedding(self, query: str) -> Embedding:
        response = self._model.create_embedding(query)
        return flatten_floats(response["data"][0]["embedding"])

    @override
    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        response = self._model.create_embedding(texts)
        return [flatten_floats(item["embedding"]) for item in response["data"]]

    @override
    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)

    @override
    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)


### Retrievers


@final
class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    @no_type_check
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    @override
    @no_type_check
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


### Workflows


@dataclass
class RAGWorkflow:
    verbose: bool = False
    fingerprint: str | None = None

    embed_model: BaseEmbedding | None = None

    llm: LLM | None = None

    storage_context: StorageContext | None = None
    vector_retriever: BaseRetriever | None = None
    keyword_retriever: BaseRetriever | None = None
    retriever: BaseRetriever | None = None

    vector_index: BaseIndex[IndexDict] | None = None
    keyword_index: BaseKeywordTableIndex | None = None

    chat_memory: ChatMemoryBuffer | ChatSummaryMemoryBuffer | None = None
    chat_history: list[ChatMessage] | None = None
    chat_engine: BaseChatEngine | None = None

    async def initialize(self, args: Args):
        if args.embed_model:
            emb = self.load_embedding(args.embed_model, args.llm_base_url)
        else:
            emb = awaitable_none()
        if args.llm:
            llm = self.load_llm(args.llm, args)
        else:
            llm = awaitable_none()
        self.embed_model = await emb
        self.llm = await llm

    async def postgres_stores(
        self, uri: str, args: Args
    ) -> tuple[PostgresDocumentStore, PostgresIndexStore, PGVectorStore]:
        logger.info("Create Postgres store objects")
        docstore: PostgresDocumentStore = PostgresDocumentStore.from_uri(
            uri=uri,
            table_name="docstore",
        )
        index_store: PostgresIndexStore = PostgresIndexStore.from_uri(
            uri=uri,
            table_name="indexstore",
        )
        vector_store: PGVectorStore = PGVectorStore.from_params(
            connection_string=uri,
            database="vector_db",
            host="localhost",
            password="",
            port=str(5432),
            user="postgres",
            table_name="vectorstore",
            embed_dim=args.embed_dim,
            hnsw_kwargs={
                "hnsw_m": args.hnsw_m,
                "hnsw_ef_construction": args.hnsw_ef_construction,
                "hnsw_ef_search": args.hnsw_ef_search,
                "hnsw_dist_method": args.hnsw_dist_method,
            },
        )
        return docstore, index_store, vector_store

    async def load_embedding(
        self, model: str, base_url: str | None = None
    ) -> BaseEmbedding:
        prefix, model = parse_prefixes(
            [
                "HuggingFace:",
                "Ollama:",
                "OpenAILike:",
                "OpenAI:",
                "LlamaCpp:",
            ],
            model,
        )
        logger.info(f"Load embedding {prefix}:{model}")
        if prefix == "HuggingFace:":
            return HuggingFaceEmbedding(model_name=model)
        elif prefix == "Ollama:":
            return OllamaEmbedding(
                model_name=model,
                base_url=base_url or "http://localhost:11434",
            )
        elif prefix == "LlamaCpp:":
            return LlamaCppEmbedding(model_path=model)
        elif prefix == "OpenAI:":
            return OpenAIEmbedding(model_name=model)
        elif prefix == "OpenAILike:":
            return OpenAILikeEmbedding(
                model_name=model,
                api_base=base_url or "http://localhost:1234/v1",
            )
        else:
            error(f"Embedding model not recognized: {model}")

    async def load_llm(self, model: str, args: Args) -> LLM:
        prefix, model = parse_prefixes(
            ["Ollama:", "OpenAILike:", "OpenAI:", "LlamaCpp:"],
            model,
        )

        logger.info(f"Load LLM {prefix}:{model}")
        if prefix == "Ollama:":
            return Ollama(
                model=model,
                base_url=args.llm_base_url or "http://localhost:11434",
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                context_window=args.context_window,
                request_timeout=args.timeout,
            )
        elif prefix == "OpenAILike:":
            return OpenAILike(
                model=model,
                api_base=args.llm_base_url or "http://localhost:1234/v1",
                api_key=args.llm_api_key,
                api_version=args.llm_api_version or "",
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                context_window=args.context_window,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
            )
        elif prefix == "OpenAI:":
            return OpenAI(
                model=model,
                api_key=args.llm_api_key,
                api_base=args.llm_base_url,
                api_version=args.llm_api_version,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                context_window=args.context_window,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
            )
        elif prefix == "LlamaCpp:":
            return LlamaCPP(
                # model_url=model_url,
                model_path=model,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
                context_window=args.context_window,
                generate_kwargs={},
                model_kwargs={"n_gpu_layers": args.gpu_layers},
                verbose=self.verbose,
            )
        else:
            error(f"LLM model not recognized: {model}")

    async def determine_fingerprint(self, input_files: list[Path]) -> str:
        logger.info("Determine input files fingerprint")
        fingerprint = [
            collection_hash(input_files),
            # hashlib.sha512(repr(args).encode("utf-8")).hexdigest(),
        ]
        final_hash = "\n".join(fingerprint).encode("utf-8")
        final_base64 = base64.b64encode(final_hash).decode("utf-8")
        return final_base64[0:32]

    async def read_documents(
        self, input_files: list[Path], num_workers: int | None
    ) -> Iterable[Document]:
        logger.info("Read documents from disk")
        file_extractor: dict[str, BaseReader] = {".org": OrgReader()}
        return SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=file_extractor,
        ).load_data(num_workers=num_workers)

    async def load_splitter(self, model: str, args: Args) -> TransformComponent:
        # Sentence
        # SentenceWindow
        # Semantic
        # Markdown
        # Code
        # Token
        # Json
        # Html
        # Hierarchical
        # Topic
        logger.info(f"Load splitter {model}")
        if model == "Sentence":
            return SentenceSplitter(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                include_metadata=True,
            )
        elif model == "SentenceWindow":
            return SentenceWindowNodeParser.from_defaults(
                window_size=args.window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
        elif model == "Semantic":
            if self.embed_model is not None:
                return SemanticSplitterNodeParser(
                    buffer_size=args.buffer_size,
                    breakpoint_percentile_threshold=args.breakpoint_percentile_threshold,
                    embed_model=self.embed_model,
                )
            else:
                error(f"Semantic splitter needs an embedding model")
        else:
            error(f"Splitting model not recognized: {model}")

    async def split_documents(
        self,
        split_model: str,
        documents: Iterable[Document],
        questions_answered: int | None,
        num_workers: int | None,
        args: Args,  # jww (2025-05-04): this should not be here
    ) -> Sequence[BaseNode]:
        logger.info(f"Split documents")

        transformations = [await self.load_splitter(split_model, args)]

        if self.llm is not None:
            transformations.extend(
                [
                    KeywordExtractor(keywords=5, llm=self.llm),
                    SummaryExtractor(summaries=["self"], llm=self.llm),
                    TitleExtractor(nodes=5, llm=self.llm),
                ]
            )
            if questions_answered is not None:
                logger.info(f"Generate {questions_answered} questions for each chunk")
                transformations.append(
                    QuestionsAnsweredExtractor(
                        questions=questions_answered, llm=self.llm
                    )
                )

        pipeline = IngestionPipeline(transformations=transformations)
        return await pipeline.arun(documents=documents, num_workers=num_workers)

    async def populate_vector_store(
        self,
        nodes: Sequence[BaseNode],
        collect_keywords: bool,
        storage_context: StorageContext,
    ) -> tuple[VectorStoreIndex, BaseKeywordTableIndex | None]:
        logger.info(f"Populate vector store")

        index_structs = storage_context.index_store.index_structs()
        for struct in index_structs:
            storage_context.index_store.delete_index_struct(key=struct.index_id)

        vector_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=self.verbose,
        )

        if collect_keywords:
            if self.llm is None:
                keyword_index = SimpleKeywordTableIndex(
                    nodes, storage_context=storage_context
                )
            else:
                keyword_index = KeywordTableIndex(
                    nodes,
                    storage_context=storage_context,
                    llm=self.llm,
                )
        else:
            keyword_index = None

        return vector_index, keyword_index

    # global Settings
    # Settings.llm = self.llm
    # Settings.embed_model = self.embed_model
    # Settings.chunk_size = args.chunk_size
    # Settings.chunk_overlap = args.chunk_overlap

    async def index_files(
        self,
        input_files: list[Path] | None,
        splitter: str,
        collect_keywords: bool,
        questions_answered: int | None,
        num_workers: int | None,
        args: Args,  # jww (2025-05-04): Should not be here
    ):
        if input_files is None:
            logger.info("No input files")
            persist_dir = None
        else:
            logger.info(f"{len(input_files)} input file(s)")
            fp: str = await self.determine_fingerprint(input_files)
            logger.info(f"Fingerprint = {fp}")
            persist_dir = cache_dir(fp)
            self.fingerprint = fp

        if args.db_conn is not None:
            docstore, index_store, vector_store = await self.postgres_stores(
                uri=args.db_conn, args=args
            )
        elif persist_dir is not None and os.path.isdir(persist_dir):
            docstore = SimpleDocumentStore.from_persist_dir(str(persist_dir))
            index_store = SimpleIndexStore.from_persist_dir(str(persist_dir))
            vector_store = SimpleVectorStore.from_persist_dir(str(persist_dir))
        else:
            docstore = SimpleDocumentStore()
            index_store = SimpleIndexStore()
            vector_store = SimpleVectorStore()

        self.storage_context = StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store,
            vector_store=vector_store,
            persist_dir=(
                str(persist_dir) if persist_dir is not None else DEFAULT_PERSIST_DIR
            ),
        )

        persisted = persist_dir is not None and os.path.isdir(persist_dir)

        if args.db_conn is not None or persisted:
            if args.db_conn is not None:
                logger.info("Read stores from database")
            else:
                logger.info("Read stores from cache")

            indices: IndexList = (  # pyright: ignore[reportUnknownVariableType]
                load_indices_from_storage(
                    storage_context=self.storage_context,
                    embed_model=self.embed_model,
                    llm=self.llm,
                )
            )
            if len(indices) == 1:
                [vector_index] = indices
                self.vector_index = vector_index
                self.keyword_index = None
            else:
                [vector_index, keyword_index] = indices
                self.vector_index = vector_index
                self.keyword_index = cast(BaseKeywordTableIndex, keyword_index)

        if input_files is not None and not persisted:
            documents = await self.read_documents(
                input_files, num_workers=args.num_workers
            )

            nodes = await self.split_documents(
                splitter,
                documents,
                questions_answered=questions_answered,
                num_workers=num_workers,
                args=args,
            )

            self.vector_index, self.keyword_index = await self.populate_vector_store(
                nodes,
                collect_keywords=collect_keywords,
                storage_context=self.storage_context,
            )

            if args.db_conn is None and persist_dir is not None:
                logger.info("Persist storage context to disk")
                self.storage_context.persist(  # pyright: ignore[reportUnknownMemberType]
                    persist_dir=persist_dir
                )

    async def load_retriever(self, similarity_top_k: int):
        if self.vector_index is not None:
            logger.info("Create vector retriever")
            self.vector_retriever = self.vector_index.as_retriever(
                similarity_top_k=similarity_top_k
            )
        if self.keyword_index is not None:
            logger.info("Create keyword retriever")
            self.keyword_retriever = self.keyword_index.as_retriever()

        if self.vector_retriever is not None:
            if self.keyword_retriever is not None:
                self.retriever = CustomRetriever(
                    vector_retriever=self.vector_retriever,
                    keyword_retriever=self.keyword_retriever,
                )
            else:
                self.retriever = self.vector_retriever
        else:
            if self.keyword_retriever is not None:
                self.retriever = self.keyword_retriever

    async def retrieve_nodes(
        self, text: str
    ) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
        if self.retriever is None:
            logger.info("No retriever")
            return []
        else:
            logger.info("Retrieve nodes from vector index")
            nodes = await self.retriever.aretrieve(text)
            logger.info(f"{len(nodes)} nodes found in vector index")
            return [{"text": node.text, "metadata": node.metadata} for node in nodes]

    # Query a document collection
    async def query(
        self,
        query: str,
        streaming: bool = False,
        retries: bool = False,
        source_retries: bool = False,
    ) -> RESPONSE_TYPE | NoReturn:
        if self.retriever is None:
            error("There is no retriever configured to query")
        if self.llm is None:
            error("There is no LLM configured to chat with")

        logger.info("Query with retriever query engine")
        query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            llm=self.llm,
            use_async=True,
            streaming=streaming,
            # response_mode=ResponseMode.REFINE,
            response_mode=ResponseMode.COMPACT,
            # response_mode=ResponseMode.SIMPLE_SUMMARIZE,
            # response_mode=ResponseMode.TREE_SUMMARIZE,
            # response_mode=ResponseMode.GENERATION, # ignore context
            # response_mode=ResponseMode.NO_TEXT, # only context
            # response_mode=ResponseMode.CONTEXT_ONLY,
            # response_mode=ResponseMode.ACCUMULATE,
            # response_mode=ResponseMode.COMPACT_ACCUMULATE,
        )

        if retries or source_retries:
            relevancy_evaluator = RelevancyEvaluator(llm=self.llm)
            # jww (2025-05-04): Allow using different evaluators
            _guideline_evaluator = GuidelineEvaluator(
                llm=self.llm,
                guidelines=DEFAULT_GUIDELINES
                + "\nThe response should not be overly long.\n"
                + "The response should try to summarize where possible.\n",
            )
            if source_retries:
                logger.info("Add retry source query engine")
                query_engine = RetrySourceQueryEngine(
                    query_engine,
                    evaluator=relevancy_evaluator,
                    llm=self.llm,
                )
            else:
                logger.info("Add retry query engine")
                query_engine = RetryQueryEngine(
                    query_engine, evaluator=relevancy_evaluator
                )

        logger.info("Submit query to LLM")
        if streaming:
            return await query_engine.aquery(query)
        else:
            return await query_engine.aquery(query)

    async def reset_chat(self):
        self.chat_engine = None
        self.chat_memory = None

    # Chat with the LLM, possibly in the context of a document collection
    async def chat(
        self,
        user: str,
        query: str,
        token_limit: int,
        chat_store: SimpleChatStore | None = None,
        summarize_chat: bool = False,
        streaming: bool = False,
    ) -> StreamingAgentChatResponse | AgentChatResponse | NoReturn:
        if self.llm is None:
            error("There is no LLM configured to chat with")

        if self.chat_memory is None and chat_store is not None:
            if summarize_chat:
                self.chat_memory = ChatSummaryMemoryBuffer.from_defaults(  # pyright: ignore[reportUnknownMemberType]
                    token_limit=token_limit,
                    # jww (2025-05-04): Make this configurable
                    summarize_prompt=(
                        "The following is a conversation between the user and assistant. "
                        "Write a concise summary about the contents of this conversation."
                    ),
                    chat_store=chat_store,
                    chat_story_key=user,
                    llm=self.llm,
                )
            else:
                self.chat_memory = ChatMemoryBuffer.from_defaults(  # pyright: ignore[reportUnknownMemberType]
                    token_limit=token_limit,
                    chat_store=chat_store,
                    chat_store_key=user,
                    llm=self.llm,
                )

        if self.chat_engine is None:
            if self.retriever is None:
                self.chat_engine = SimpleChatEngine.from_defaults(
                    llm=self.llm,
                    memory=self.chat_memory,
                    system_prompt="You are a helpful AI assistant.",
                )
            else:
                self.chat_engine = ContextChatEngine.from_defaults(
                    retriever=self.retriever,
                    llm=self.llm,
                    memory=self.chat_memory,
                    system_prompt="You are a helpful AI assistant.",
                )

        logger.info("Submit chat to LLM")
        if streaming:
            return await self.chat_engine.astream_chat(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                message=query
            )
        else:
            return await self.chat_engine.achat(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                message=query
            )


### Main


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


async def rag_initialize(args: Args) -> RAGWorkflow:
    rag = RAGWorkflow(verbose=args.verbose)
    await rag.initialize(args)

    if args.from_:
        input_files = read_files(args.from_, args.recursive)
    else:
        input_files = None

    await rag.index_files(
        input_files,
        splitter=args.splitter,
        collect_keywords=args.collect_keywords,
        questions_answered=args.questions_answered,
        num_workers=args.num_workers,
        args=args,
    )

    await rag.load_retriever(similarity_top_k=args.top_k)
    return rag


async def rag_client(args: Args):
    rag: RAGWorkflow = await rag_initialize(args)
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


def rebuild_postgres_db(db_name: str):
    connection_string = "postgresql://postgres:password@localhost:5432"
    with psycopg2.connect(connection_string) as conn:
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {db_name}")
            c.execute(f"CREATE DATABASE {db_name}")


def parse_args(
    arguments: list[str] = sys.argv[1:], config_path: Path | None = None
) -> Args:
    parser = argparse.ArgumentParser()

    _ = parser.add_argument(
        "--config", "-c", type=str, help="Yaml config file, to set argument defaults"
    )
    _ = parser.add_argument("--verbose", action="store_true", help="Verbose?")
    _ = parser.add_argument(
        "--db_conn",
        type=str,
        help="Postgres connection string (in-memory if unspecified)",
    )
    _ = parser.add_argument(
        "--hnsw_m",
        type=int,
        default=16,
        help="Bi-dir links for each node (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--hnsw_ef_construction",
        type=int,
        default=64,
        help="Dynamic candidate list size (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--hnsw_ef_search",
        type=int,
        default=40,
        help="Candidate list size during search (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--hnsw_dist_method",
        type=str,
        default="vector_cosine_ops",
        help="Distance method for similarity (default: %(default)s)",
    )
    _ = parser.add_argument("--embed_model", type=str, help="Embedding model")
    _ = parser.add_argument(
        "--embed_base_url",
        type=str,
        help="URL to use for talking with embedding model",
    )
    _ = parser.add_argument(
        "--embed_dim",
        type=int,
        default=512,
        help="Embedding dimensions (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--chunk_size", type=int, default=512, help="Chunk size (default: %(default)s)"
    )
    _ = parser.add_argument(
        "--chunk_overlap",
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
        "--buffer_size",
        type=int,
        default=256,
        help="Buffer size for semantic splitting (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--breakpoint_percentile_threshold",
        type=int,
        default=95,
        help="Breakpoint percentile threshold (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--window_size",
        type=int,
        default=3,
        help="Window size of sentence window splitter (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--questions_answered",
        type=int,
        help="If provided, generate N questions related to each chunk",
    )
    _ = parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top K document nodes (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--llm",
        type=str,
        help="LLM to use for text generation and chat",
    )
    _ = parser.add_argument(
        "--llm_api_key",
        type=str,
        default="fake",
        help="API key to use with LLM",
    )
    _ = parser.add_argument(
        "--llm_api_version",
        type=str,
        help="API version to use with LLM (if required)",
    )
    _ = parser.add_argument(
        "--llm_base_url",
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
        "--max_tokens",
        type=int,
        default=200,
        help="LLM maximum answer size in tokens (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--context_window",
        type=int,
        default=8192,
        help="LLM context window size (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--reasoning_effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="LLM reasoning effort (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--gpu_layers",
        type=int,
        default=-1,
        help="Number of GPU layers to use (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--chat_user",
        type=str,
        help="Chat user name for history saves (no history if unset)",
    )
    _ = parser.add_argument(
        "--token_limit",
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
        "--collect_keywords",
        action="store_true",
        help="Generate keywords for document retrieval",
    )
    _ = parser.add_argument(
        "--retries",
        action="store_true",
        help="Retry queries based on relevancy",
    )
    _ = parser.add_argument(
        "--source_retries",
        action="store_true",
        help="Retry queries (using source modification) based on relevancy",
    )
    _ = parser.add_argument(
        "--summarize_chat",
        action="store_true",
        help="Summarize chat history when it grows too long",
    )
    _ = parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of works to use for various tasks (default: %(default)s)",
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
    logging.basicConfig(
        stream=sys.stdout,
        encoding="utf-8",
        level=logging.INFO if args.verbose else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",  # Log message format
        datefmt="%H:%M:%S",
    )
    asyncio.run(rag_client(args))


if __name__ == "__main__":
    main(parse_args())

# store
# llm
# files
# fingerprint
# cache
# embedding
# store_index
# context
# read
# split
# transform
# index
# search
# query
# chat

# from llama_index.core.tools import FunctionTool
# from llama_index.llms.openai import OpenAI


# def add(x: int, y: int) -> int:
#     """Useful function to add two numbers."""
#     return x + y


# def multiply(x: int, y: int) -> int:
#     """Useful function to multiply two numbers."""
#     return x * y


# tools = [
#     FunctionTool.from_defaults(add),
#     FunctionTool.from_defaults(multiply),
# ]

# agent = ReActAgent(
#     llm=OpenAI(model="gpt-4o"), tools=tools, timeout=120, verbose=True
# )

# ret = await agent.run(input="Hello!")
