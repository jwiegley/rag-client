#!/usr/bin/env python

import asyncio
import base64
import hashlib
import json
import logging
import os
import sys
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cache
from io import StringIO
from pathlib import Path
from typing import Any, Literal, NoReturn, cast, final, no_type_check, override

from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
import typed_argparse as tap
from llama_index.core import (
    KeywordTableIndex,
    QueryBundle,
    Settings,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.indices import (
    load_index_from_storage,  # pyright: ignore[reportUnknownVariableType]
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.keyword_table.base import BaseKeywordTableIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from llama_index.core.node_parser import (
    NodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
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
from llama_index.core.vector_stores.types import BasePydanticVectorStore
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
from orgparse.node import OrgNode
from typed_argparse import TypedArgs, arg
from xdg_base_dirs import xdg_cache_home

os.environ["TOKENIZERS_PARALLELISM"] = "false"

### Utility functions


logger = logging.getLogger("rag-client")


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
    verbose: bool = arg(help="Verbose?")
    db_name: str | None = arg(
        help="Postgres db (in-memory vector index if unspecified)",
    )
    db_host: str = arg(
        default="localhost",
        help="Postgres db host (default: %(default)s)",
    )
    db_port: int = arg(
        default=5432,
        help="Postgres db port (default: %(default)s)",
    )
    db_user: str = arg(
        default="postgres",
        help="Postgres db user (default: %(default)s)",
    )
    db_pass: str = arg(default="", help="Postgres db password")
    db_table: str = arg(
        default="vectors",
        help="Postgres db table (default: %(default)s)",
    )
    hnsw_m: int = arg(
        default=16,
        help="Bi-dir links for each node (default: %(default)s)",
    )
    hnsw_ef_construction: int = arg(
        default=64,
        help="Dynamic candidate list size (default: %(default)s)",
    )
    hnsw_ef_search: int = arg(
        default=40,
        help="Candidate list size during search (default: %(default)s)",
    )
    hnsw_dist_method: str = arg(
        default="vector_cosine_ops",
        help="Distance method for similarity (default: %(default)s)",
    )
    embed_model: str | None = arg(help="Embedding model")
    embed_base_url: str | None = arg(
        help="URL to use for talking with embedding model",
    )
    embed_dim: int = arg(
        default=512,
        help="Embedding dimensions (default: %(default)s)",
    )
    chunk_size: int = arg(default=512, help="Chunk size (default: %(default)s)")
    chunk_overlap: int = arg(
        default=20,
        help="Chunk overlap (default: %(default)s)",
    )
    questions_answered: int | None = arg(
        help="If provided, generate N questions related to each chunk",
    )
    top_k: int = arg(
        default=3,
        help="Top K document nodes (default: %(default)s)",
    )
    llm: str | None = arg(
        help="LLM to use for text generation and chat",
    )
    llm_api_key: str = arg(
        default="fake",
        help="API key to use with LLM",
    )
    llm_api_version: str | None = arg(
        help="API version to use with LLM (if required)",
    )
    llm_base_url: str | None = arg(
        help="URL to use for talking with LLM (default depends on LLM)",
    )
    streaming: bool = arg(
        help="Stream output as it arrives from LLM",
    )
    timeout: int = arg(
        default=60,
        help="Max time to wait in seconds (default: %(default)s)",
    )
    temperature: float = arg(
        default=1.0,
        help="LLM temperature value (default: %(default)s)",
    )
    max_tokens: int = arg(
        default=256,
        help="LLM maximum answer size in tokens (default: %(default)s)",
    )
    context_window: int = arg(
        default=8192,
        help="LLM context window size (default: %(default)s)",
    )
    reasoning_effort: Literal["low", "medium", "high"] = arg(
        default="medium",
        help="LLM reasoning effort (default: %(default)s)",
    )
    gpu_layers: int = arg(
        default=-1,
        help="Number of GPU layers to use (default: %(default)s)",
    )
    token_limit: int = arg(
        default=1500,
        help="Token limit used for chat history (default: %(default)s)",
    )
    from_: str | None = arg("--from", help="Where to read files from (optional)")
    recursive: bool = arg(
        help="Read directories recursively (default: no)",
    )
    use_keywords: bool = arg(
        help="Generate keywords for document retrieval",
    )
    retries: bool = arg(
        help="Retry queries based on relevancy",
    )
    source_retries: bool = arg(
        help="Retry queries (using source modification) based on relevancy",
    )
    summarize_chat: bool = arg(
        help="Summarize chat history when it grows too long",
    )
    command: str = arg(positional=True, help="Command to execute")
    args: list[str] = arg(positional=True, nargs="*", help="Query to submit to LLM")


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


### VectorStoreObjects


@dataclass
class VectorStoreObject:
    @abstractmethod
    def construct(self) -> BasePydanticVectorStore:
        pass


@dataclass
class PGVectorStoreObject(VectorStoreObject):
    database: str
    host: str
    password: str
    port: int
    user: str
    table_name: str
    embed_dim: int
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    hnsw_dist_method: str

    @override
    def construct(self) -> BasePydanticVectorStore:
        logger.info("Setup PostgreSQL vector store")
        return PGVectorStore.from_params(
            database=self.database,
            host=self.host,
            password=self.password,
            port=str(self.port),
            user=self.user,
            table_name=self.table_name,
            embed_dim=self.embed_dim,
            hnsw_kwargs={
                "hnsw_m": self.hnsw_m,
                "hnsw_ef_construction": self.hnsw_ef_construction,
                "hnsw_ef_search": self.hnsw_ef_search,
                "hnsw_dist_method": self.hnsw_dist_method,
            },
            # create_engine_kwargs={
            #     "connect_args": {"options": "-c client_encoding=UTF8"}
            # },
        )


### EmbeddingObjects


@dataclass
class EmbeddingObject:
    base_url: str | None

    def construct(self, model: str) -> BaseEmbedding | NoReturn:
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
                base_url=self.base_url or "http://localhost:11434",
            )
        elif prefix == "LlamaCpp:":
            return LlamaCppEmbedding(model_path=model)
        elif prefix == "OpenAI:":
            return OpenAIEmbedding(model_name=model)
        elif prefix == "OpenAILike:":
            return OpenAILikeEmbedding(
                model_name=model,
                api_base=self.base_url or "http://localhost:1234/v1",
            )
        else:
            error(f"Embedding model not recognized: {model}")


### SplitterObjects


@dataclass
class SplitterObject:
    chunk_size: int
    chunk_overlap: int

    embed_model: BaseEmbedding | None
    buffer_size: int
    breakpoint_percentile_threshold: int

    def construct(self, model: str) -> NodeParser | NoReturn:
        prefix, model = parse_prefixes(
            [
                "Sentence:",
                "SentenceWindow:",
                "Semantic:",
                "Markdown:",
                "Code:",
                "Token:",
                "Json:",
                "Html:",
                "Hierarchical:",
                "Topic:",
            ],
            model,
        )
        logger.info(f"Load splitter {prefix}:{model}")
        if prefix == "Sentence:":
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                include_metadata=True,
            )
        elif prefix == "Semantic:":
            if self.embed_model is not None:
                # jww (2025-05-04): Make this configurable
                return SemanticSplitterNodeParser(
                    buffer_size=self.buffer_size,
                    breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
                    embed_model=self.embed_model,
                )
            else:
                error(f"Semantic splitter needs an embedding model")
        else:
            error(f"Splitting model not recognized: {model}")


### LLMObjects


@dataclass
class LLMObject:
    base_url: str | None
    api_key: str
    api_version: str | None
    temperature: float
    max_tokens: int
    context_window: int
    reasoning_effort: Literal["low", "medium", "high"]
    timeout: int
    gpu_layers: int

    def construct(self, model: str, verbose: bool = False) -> LLM | NoReturn:
        prefix, model = parse_prefixes(
            ["Ollama:", "OpenAILike:", "OpenAI:", "LlamaCpp:"],
            model,
        )

        logger.info(f"Load LLM {prefix}:{model}")
        if prefix == "Ollama:":
            return Ollama(
                model=model,
                base_url=self.base_url or "http://localhost:11434",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                context_window=self.context_window,
                request_timeout=self.timeout,
            )
        elif prefix == "OpenAILike:":
            return OpenAILike(
                model=model,
                api_base=self.base_url or "http://localhost:1234/v1",
                api_key=self.api_key,
                api_version=self.api_version or "",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                context_window=self.context_window,
                reasoning_effort=self.reasoning_effort,
                timeout=self.timeout,
            )
        elif prefix == "OpenAI:":
            return OpenAI(
                model=model,
                api_key=self.api_key,
                api_base=self.base_url,
                api_version=self.api_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                context_window=self.context_window,
                reasoning_effort=self.reasoning_effort,
                timeout=self.timeout,
            )
        elif prefix == "LlamaCpp:":
            return LlamaCPP(
                # model_url=model_url,
                model_path=model,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                context_window=self.context_window,
                generate_kwargs={},
                model_kwargs={"n_gpu_layers": self.gpu_layers},
                verbose=verbose,
            )
        else:
            error(f"LLM model not recognized: {model}")


### Workflows


@dataclass
class RAGWorkflow:
    args: Args

    fingerprint: str | None = None

    vector_store: BasePydanticVectorStore = field(default_factory=SimpleVectorStore)
    embed_model: BaseEmbedding | None = None

    llm: LLM | None = None

    vector_retriever: BaseRetriever | None = None
    keyword_retriever: BaseRetriever | None = None
    retriever: BaseRetriever | None = None

    vector_index: BaseIndex[IndexDict] | None = None
    keyword_index: BaseKeywordTableIndex | None = None

    chat_memory: ChatMemoryBuffer | ChatSummaryMemoryBuffer | None = None
    chat_history: list[ChatMessage] | None = None
    chat_engine: BaseChatEngine | None = None

    async def initialize(self):
        self.vector_store, self.embed_model, self.llm, input_files = (
            await asyncio.gather(
                self.load_vector_store(),
                self.load_embedding(),
                self.load_llm(),
                self.find_input_files(),
            )
        )
        await self.load_vector_index(input_files)

    async def load_vector_store(self):
        args: Args = self.args
        if args.db_name is None:
            logger.info("In-memory vector store")
            return SimpleVectorStore()
        else:
            logger.info("Postgres vector store")
            return PGVectorStoreObject(
                database=args.db_name,
                host=args.db_host,
                password=args.db_pass,
                port=args.db_port,
                user=args.db_user,
                table_name=args.db_table,
                embed_dim=args.embed_dim,
                hnsw_m=args.hnsw_m,
                hnsw_ef_construction=args.hnsw_ef_construction,
                hnsw_ef_search=args.hnsw_ef_search,
                hnsw_dist_method=args.hnsw_dist_method,
            ).construct()

    async def load_embedding(self):
        args: Args = self.args
        if args.embed_model is None:
            logger.info("No embedding model")
            return None
        else:
            return EmbeddingObject(base_url=args.embed_base_url).construct(
                model=args.embed_model
            )

    async def load_llm(self):
        args: Args = self.args
        if args.llm is None:
            logger.info("No LLM")
            return None
        else:
            return LLMObject(
                base_url=args.llm_base_url,
                api_key=args.llm_api_key,
                api_version=args.llm_api_version,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                context_window=args.context_window,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
                gpu_layers=args.gpu_layers,
            ).construct(args.llm, verbose=args.verbose)

    async def find_input_files(self) -> list[Path] | None:
        args: Args = self.args
        if args.from_ is None:
            logger.info("No input files")
            return None
        else:
            input_files = read_files(args.from_, args.recursive)
            return input_files

    async def determine_fingerprint(self, input_files: list[Path]) -> str:
        args = deepcopy(self.args)
        args.command = ""
        args.args = []
        fingerprint = [
            collection_hash(input_files),
            hashlib.sha512(repr(args).encode("utf-8")).hexdigest(),
        ]
        final_hash = "\n".join(fingerprint).encode("utf-8")
        final_base64 = base64.b64encode(final_hash).decode("utf-8")
        return final_base64[0:32]

    def read_documents(self, input_files: list[Path]) -> Iterable[Document]:
        logger.info("Read documents from disk")

        file_extractor: dict[str, BaseReader] = {".org": OrgReader()}

        # jww (2025-05-04): Make num_workers configurable
        return SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=file_extractor,
        ).load_data(num_workers=4)

    async def split_documents(
        self, documents: Iterable[Document]
    ) -> Sequence[BaseNode]:
        args: Args = self.args
        logger.info(f"Split documents")

        transformations: list[TransformComponent] = [
            SentenceSplitter(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                include_metadata=True,
            ),
            # jww (2025-05-04): Semantic-aware splitting
            # SemanticSplitterNodeParser(
            #     buffer_size=256,
            #     breakpoint_percentile_threshold=95
            # )
        ]

        if self.llm is not None and args.questions_answered is not None:
            logger.info(f"Generate {args.questions_answered} questions for each chunk")
            transformations.append(
                QuestionsAnsweredExtractor(
                    questions=args.questions_answered, llm=self.llm
                )
            )

        pipeline = IngestionPipeline(transformations=transformations)
        # jww (2025-05-03): Make num_workers here customizable
        return await pipeline.arun(documents=documents, num_workers=4)

    async def populate_vector_store(
        self, nodes: Sequence[BaseNode]
    ) -> tuple[VectorStoreIndex, BaseKeywordTableIndex | None]:
        args: Args = self.args

        logger.info("Create storage context")
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        logger.info(f"Populate vector store")
        vector_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=args.verbose,
        )

        if args.use_keywords:
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

    async def load_index_from_cache(
        self, persist_dir: Path, index_id: str
    ) -> BaseIndex[IndexDict]:
        args: Args = self.args
        logger.info("Load index from cache")
        global Settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = args.chunk_size
        Settings.chunk_overlap = args.chunk_overlap
        return load_index_from_storage(  # pyright: ignore[reportUnknownVariableType]
            StorageContext.from_defaults(persist_dir=str(persist_dir)),
            index_id=index_id,
        )

    async def load_index_from_vector_store(self) -> VectorStoreIndex:
        args: Args = self.args
        logger.info("Load index from database")
        return VectorStoreIndex.from_vector_store(  # pyright: ignore[reportUnknownMemberType]
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            show_progress=args.verbose,
        )

    async def load_vector_index(self, input_files: list[Path] | None):
        args: Args = self.args
        if input_files is None:
            logger.info("No input files")
            if not isinstance(self.vector_store, SimpleVectorStore):
                self.vector_index = await self.load_index_from_vector_store()
        else:
            logger.info(f"{len(input_files)} input file(s)")
            fp: str = await self.determine_fingerprint(input_files)
            logger.info(f"Fingerprint = {fp}")
            persist_dir = cache_dir(fp)
            if (self.fingerprint is None or self.fingerprint == fp) and os.path.isdir(
                persist_dir
            ):
                vector_index, keyword_index = await asyncio.gather(
                    self.load_index_from_cache(persist_dir / "v", "vectors"),
                    self.load_index_from_cache(persist_dir / "k", "keywords"),
                )
                self.vector_index = vector_index
                self.keyword_index = cast(BaseKeywordTableIndex, keyword_index)
            else:
                documents = self.read_documents(input_files)
                nodes = await self.split_documents(documents)
                self.vector_index, self.keyword_index = (
                    await self.populate_vector_store(nodes)
                )

                if isinstance(self.vector_store, SimpleVectorStore):
                    logger.info("Write index to cache")
                    self.vector_index.storage_context.persist(  # pyright: ignore[reportUnknownMemberType]
                        persist_dir=(persist_dir / "v")
                    )
                if self.keyword_index is not None:
                    self.keyword_index.storage_context.persist(  # pyright: ignore[reportUnknownMemberType]
                        persist_dir=persist_dir / "k"
                    )
            self.fingerprint = fp

        logger.info("Create retriever object")
        if self.vector_index is not None:
            self.vector_retriever = self.vector_index.as_retriever(
                similarity_top_k=args.top_k
            )
        if self.keyword_index is not None:
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
            return []
        else:
            logger.info("Retrieve nodes from vector index")
            nodes = await self.retriever.aretrieve(text)
            logger.info(f"{len(nodes)} nodes found in vector index")
            return [{"text": node.text, "metadata": node.metadata} for node in nodes]

    # Query a document collection
    @no_type_check
    async def query(
        self,
        query: str,
        streaming: bool = False,
        print_responses: bool = False,
    ) -> str | NoReturn:
        args = self.args
        if self.retriever is None:
            error("There is no retriever configured to query")
        if self.llm is None:
            error("There is no LLM configured to chat with")

        query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            llm=self.llm,
            use_async=True,
            streaming=streaming,
        )
        if args.retries or args.source_retries:
            query_response_evaluator = RelevancyEvaluator(llm=self.llm)
            # jww (2025-05-04): Allow using different evaluators
            _guideline_evaluator = GuidelineEvaluator(
                guidelines=DEFAULT_GUIDELINES
                + "\nThe response should not be overly long.\n"
                + "The response should try to summarize where possible.\n"
            )
            if args.source_retries:
                query_engine = RetrySourceQueryEngine(
                    query_engine,
                    evaluator=query_response_evaluator,
                    llm=self.llm,
                )
            else:
                query_engine = RetryQueryEngine(
                    query_engine, evaluator=query_response_evaluator
                )

        if streaming:
            response = await query_engine.aquery(query)
            full_response = StringIO()
            async for token in response.response_gen:
                token = clean_special_tokens(token)
                if print_responses:
                    print(token, end="", flush=True)
                _ = full_response.write(token)
            return full_response.getvalue()
        else:
            response = await query_engine.aquery(query)
            if print_responses:
                print(str(response))
            return clean_special_tokens(str(response))

    # Chat with the LLM, possibly in the context of a document collection
    @no_type_check
    async def chat(
        self,
        query: str,
        keep_history: bool = True,
        streaming: bool = False,
        print_responses: bool = False,
    ) -> str | NoReturn:
        args = self.args
        if self.llm is None:
            error("There is no LLM configured to chat with")

        chat_store = SimpleChatStore()
        # chat_store = PostgresChatStore.from_uri(
        #     uri="postgresql+asyncpg://postgres:password@127.0.0.1:5432/database",
        # )

        if self.chat_memory is None and keep_history:
            if args.summarize_chat:
                self.chat_memory = ChatSummaryMemoryBuffer.from_defaults(
                    token_limit=args.token_limit,
                    # jww (2025-05-04): Make this configurable
                    summarize_prompt=(
                        "The following is a conversation between the user and assistant. "
                        "Write a concise summary about the contents of this conversation."
                    ),
                    chat_store=chat_store,
                    chat_story_key="user1",
                    llm=self.llm,
                )
            else:
                self.chat_memory = ChatMemoryBuffer.from_defaults(
                    token_limit=args.token_limit,
                    chat_store=chat_store,
                    chat_store_key="user1",
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

        if streaming:
            generator = await self.chat_engine.astream_chat(message=query)
            full_response = StringIO()
            async for token in generator.async_response_gen():
                token = clean_special_tokens(token)
                if print_responses:
                    print(token, end="", flush=True)
                _ = full_response.write(token)
            return full_response.getvalue()
        else:
            response = await self.chat_engine.achat(message=query)
            if print_responses:
                print(str(response))
            return clean_special_tokens(str(response))

    async def search_command(self, query: str):
        nodes = await self.retrieve_nodes(query)
        print(json.dumps(nodes, indent=2))

    @no_type_check
    async def query_command(self, query: str):
        args = self.args
        print(
            await self.query(
                query,
                streaming=args.streaming,
                print_responses=True,
            )
        )

    @no_type_check
    async def chat_command(self):
        args = self.args
        while True:
            query = input("\nUSER> ")
            if query == "exit":
                break
            elif query.startswith("search "):
                await self.search_command(query[7:])
            elif query.startswith("query "):
                await self.query_command(query[6:])
            else:
                _ = await self.chat(
                    query,
                    keep_history=True,
                    streaming=args.streaming,
                    print_responses=True,
                )
                print()

    @no_type_check
    async def execute(self, command: str, args: list[str]):
        match command:
            case "search":
                await self.search_command(args[0])
            case "query":
                await self.query_command(args[0])
            case "chat":
                await self.chat_command()
            case _:
                error(f"Command unrecognized: {command}")


### Main


@no_type_check
async def rag_client(args: Args):
    rag_workflow = RAGWorkflow(args)
    await rag_workflow.initialize()
    await rag_workflow.execute(args.command, args.args)


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
    tap.Parser(Args).bind(main).run()

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
