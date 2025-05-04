#!/usr/bin/env python

import asyncio
import base64
from collections.abc import Sequence
import hashlib
import json
import logging
import os
import sys
import typed_argparse as tap

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, NoReturn, cast, no_type_check, override
from typed_argparse import TypedArgs, arg
from xdg_base_dirs import xdg_cache_home
from orgparse.node import OrgNode

from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.indices import (
    load_index_from_storage,  # pyright: ignore[reportUnknownVariableType]
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import (
    BaseNode,
    Document,
    Node,
    TransformComponent,
)
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,  # pyright: ignore[reportUnknownVariableType]
)

# from llama_index.utils.workflow import draw_most_recent_execution

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


### Events


class ProgressEvent(Event):
    msg: str


class CommandEvent(Event):
    history: list[str]


class HaveSearchCommand(Event):
    pass


class HaveQueryCommand(Event):
    pass


class HaveChatCommand(Event):
    pass


class SearchInputEvent(Event):
    text: str


class SearchResultEvent(Event):
    nodes: list[dict[str, Any]]  # pyright: ignore[reportExplicitAny]


class QueryInputEvent(Event):
    query: str
    history: list[str]


class QueryBuiltEvent(Event):
    user_string: str
    query: str
    history: list[str]


class QueryResponseEvent(Event):
    response: str
    history: list[str]


class HaveVectorStoreEvent(Event):
    vector_store: BasePydanticVectorStore


class PopulatedVectorStoreEvent(Event):
    vector_store: BasePydanticVectorStore


class HaveVectorIndexEvent(Event):
    vector_index: BaseIndex[IndexDict]


class HaveLLMEvent(Event):
    llm: LLM


class HaveEmbeddingEvent(Event):
    embed_model: BaseEmbedding


class HaveInputFilesEvent(Event):
    input_files: list[Path]


class HaveFingerprintEvent(Event):
    fingerprint: str


class HavePersistDirEvent(Event):
    persist_dir: Path


class ExistingPersistDirEvent(Event):
    persist_dir: Path


class HaveDocumentsEvent(Event):
    documents: list[Document]


class TransformedNodesEvent(Event):
    nodes: Sequence[BaseNode]


class InMemoryVectorStoreEvent(Event):
    pass


class NoInMemoryVectorStoreEvent(Event):
    pass


class NoEmbeddingEvent(Event):
    pass


class NoLLMEvent(Event):
    pass


class NoInputFilesEvent(Event):
    pass


class NoVectorIndexEvent(Event):
    pass


class NoExistingPersistDirEvent(Event):
    pass


### Workflows


class RAGWorkflow(Workflow):
    @step
    async def rag_workflow(self, _ev: StartEvent) -> CommandEvent:
        return CommandEvent(history=[])

    @step
    async def process_command(
        self, ev: CommandEvent, ctx: Context
    ) -> (
        SearchInputEvent
        | QueryInputEvent
        | HaveSearchCommand
        | HaveQueryCommand
        | HaveChatCommand
        | StopEvent
    ):
        args: Args = await ctx.get("args")
        history: list[str] = ev.history
        match args.command:
            case "search":
                ctx.send_event(HaveSearchCommand())
                return SearchInputEvent(text=args.args[0])
            case "query":
                ctx.send_event(HaveQueryCommand())
                ctx.send_event(SearchInputEvent(text=args.args[0]))
                return QueryInputEvent(query=args.args[0], history=history)
            case "chat":
                print("[USER]")
                query = input("> ")
                ctx.send_event(HaveChatCommand())
                ctx.send_event(SearchInputEvent(text=query))
                return QueryInputEvent(query=query, history=history)
            case _:
                return StopEvent(f"Command unrecognized: {args.command}")

    @step
    async def load_vector_store(
        self, _ev: StartEvent, ctx: Context
    ) -> HaveVectorStoreEvent | InMemoryVectorStoreEvent | NoInMemoryVectorStoreEvent:
        args: Args = await ctx.get("args")
        if args.db_name is None:
            logger.info("In-memory vector store")
            ctx.send_event(InMemoryVectorStoreEvent())
            return HaveVectorStoreEvent(vector_store=SimpleVectorStore())
        else:
            logger.info("Postgres vector store")
            ctx.send_event(NoInMemoryVectorStoreEvent())
            vector_store = PGVectorStoreObject(
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
            return HaveVectorStoreEvent(vector_store=vector_store)

    @step
    async def load_embedding(
        self, _ev: StartEvent, ctx: Context
    ) -> HaveEmbeddingEvent | None:
        args: Args = await ctx.get("args")
        if args.embed_model is None:
            logger.info("No embedding model")
            return None
        else:
            embed_model = EmbeddingObject(base_url=args.embed_base_url).construct(
                model=args.embed_model
            )
            return HaveEmbeddingEvent(embed_model=embed_model)

    @step
    async def load_llm(
        self, _ev: StartEvent, ctx: Context
    ) -> HaveLLMEvent | NoLLMEvent:
        args: Args = await ctx.get("args")
        if args.llm is None:
            logger.info("No LLM")
            return NoLLMEvent()
        else:
            llm = LLMObject(
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
            return HaveLLMEvent(llm=llm)

    @step
    async def find_input_files(
        self, _ev: StartEvent, ctx: Context
    ) -> HaveInputFilesEvent | NoInputFilesEvent | StopEvent:
        args: Args = await ctx.get("args")
        if args.from_ is None:
            logger.info("No input files")
            if args.command == "files":
                return StopEvent([])
            else:
                return NoInputFilesEvent()
        else:
            input_files = read_files(args.from_, args.recursive)
            if args.command == "files":
                return StopEvent(input_files)
            else:
                return HaveInputFilesEvent(input_files=input_files)

    @step
    async def determine_fingerprint(
        self, ev: HaveInputFilesEvent, ctx: Context
    ) -> HaveFingerprintEvent:
        args: Args = await ctx.get("args")
        input_files: list[Path] = ev.input_files
        fingerprint = [
            collection_hash(input_files),
            hashlib.sha512(str(args.chunk_size).encode("utf-8")).hexdigest(),
            hashlib.sha512(str(args.chunk_overlap).encode("utf-8")).hexdigest(),
        ]
        if args.embed_model is not None:
            fingerprint.append(
                hashlib.sha512(args.embed_model.encode("utf-8")).hexdigest()
            )
        final_hash = "\n".join(fingerprint).encode("utf-8")
        final_base64 = base64.b64encode(final_hash).decode("utf-8")
        return HaveFingerprintEvent(fingerprint=final_base64[0:32])

    @step
    async def locate_persist_dir(
        self, ev: HaveFingerprintEvent, ctx: Context
    ) -> ExistingPersistDirEvent | HavePersistDirEvent | NoExistingPersistDirEvent:
        fingerprint: str = ev.fingerprint
        persist_dir = cache_dir(fingerprint)
        logger.info(f"Cache directory = {persist_dir}")
        if os.path.isdir(persist_dir):
            return ExistingPersistDirEvent(persist_dir=persist_dir)
        else:
            ctx.send_event(NoExistingPersistDirEvent())
            return HavePersistDirEvent(persist_dir=persist_dir)

    @step
    async def load_index_from_cache(
        self, ev: ExistingPersistDirEvent | HaveEmbeddingEvent, ctx: Context
    ) -> HaveVectorIndexEvent | None:
        data = ctx.collect_events(ev, [ExistingPersistDirEvent, HaveEmbeddingEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        persist_dir_event, embedding_event = data
        persist_dir_event = cast(ExistingPersistDirEvent, persist_dir_event)
        embedding_event = cast(HaveEmbeddingEvent, embedding_event)

        args: Args = await ctx.get("args")
        logger.info("Load index from cache")
        global Settings
        embed_model: BaseEmbedding = embedding_event.embed_model
        Settings.embed_model = embed_model
        Settings.chunk_size = args.chunk_size
        Settings.chunk_overlap = args.chunk_overlap
        persist_dir: Path = persist_dir_event.persist_dir
        vector_index: BaseIndex[IndexDict] = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(persist_dir))
        )
        return HaveVectorIndexEvent(vector_index=vector_index)

    @step
    async def load_index_from_vector_store(
        self,
        ev: (
            HaveVectorStoreEvent
            | HaveEmbeddingEvent
            | NoInputFilesEvent
            | NoInMemoryVectorStoreEvent
        ),
        ctx: Context,
    ) -> HaveVectorIndexEvent | None:
        data = ctx.collect_events(
            ev,
            [
                HaveVectorStoreEvent,
                HaveEmbeddingEvent,
                NoInputFilesEvent,
                NoInMemoryVectorStoreEvent,
            ],
        )
        if data is None:
            return None  # Not all required events have arrived yet
        vector_store_event, embedding_event, _, _ = data
        vector_store_event = cast(HaveVectorStoreEvent, vector_store_event)
        embedding_event = cast(HaveEmbeddingEvent, embedding_event)

        vector_store = vector_store_event.vector_store

        args: Args = await ctx.get("args")
        logger.info("Load index from database")
        embed_model: BaseEmbedding = embedding_event.embed_model
        vector_index: VectorStoreIndex = (
            VectorStoreIndex.from_vector_store(  # pyright: ignore[reportUnknownMemberType]
                vector_store=vector_store,
                embed_model=embed_model,
                show_progress=args.verbose,
            )
        )
        return HaveVectorIndexEvent(vector_index=vector_index)

    @step
    async def read_documents(
        self, ev: HaveInputFilesEvent | NoExistingPersistDirEvent, ctx: Context
    ) -> HaveDocumentsEvent | None:
        data = ctx.collect_events(ev, [HaveInputFilesEvent, NoExistingPersistDirEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        input_files_event, _ = data
        input_files_event = cast(HaveInputFilesEvent, input_files_event)

        file_extractor: dict[str, BaseReader] = {".org": OrgReader()}

        logger.info("Read documents from disk")
        input_files: list[Path] = input_files_event.input_files
        documents = SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=file_extractor,
        ).load_data()
        return HaveDocumentsEvent(documents=documents)

    @step
    async def split_documents_without_llm(
        self, ev: HaveDocumentsEvent | NoLLMEvent, ctx: Context
    ) -> TransformedNodesEvent | None:
        data = ctx.collect_events(ev, [HaveDocumentsEvent, NoLLMEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        documents_event, no_llm_event = data
        documents_event = cast(HaveDocumentsEvent, documents_event)
        no_llm_event = cast(NoLLMEvent, no_llm_event)

        # Whenever we consume the NoLLMEvent, we must generate it again so
        # that other steps can also use it
        ctx.send_event(no_llm_event)

        args: Args = await ctx.get("args")
        documents: list[Document] = documents_event.documents

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    include_metadata=True,
                ),
                # # Semantic-aware splitting
                # SemanticSplitterNodeParser(
                #     buffer_size=256,
                #     breakpoint_percentile_threshold=95
                # )
            ]
        )
        # jww (2025-05-03): Make num_workers here customizable
        nodes: Sequence[BaseNode] = await pipeline.arun(
            documents=documents, num_workers=4
        )
        return TransformedNodesEvent(nodes=nodes)

    @step
    async def split_documents_with_llm(
        self, ev: HaveDocumentsEvent | HaveLLMEvent, ctx: Context
    ) -> TransformedNodesEvent | None:
        data = ctx.collect_events(ev, [HaveDocumentsEvent, HaveLLMEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        documents_event, llm_event = data
        documents_event = cast(HaveDocumentsEvent, documents_event)
        llm_event = cast(HaveLLMEvent, llm_event)

        # Whenever we consume the HaveLLMEvent, we must generate it again so
        # that other steps can also use it
        ctx.send_event(llm_event)

        llm: LLM = llm_event.llm

        args: Args = await ctx.get("args")
        documents: list[Document] = documents_event.documents

        transformations: list[TransformComponent] = [
            SentenceSplitter(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                include_metadata=True,
            ),
            # # Semantic-aware splitting
            # SemanticSplitterNodeParser(
            #     buffer_size=256,
            #     breakpoint_percentile_threshold=95
            # )
        ]

        if args.questions_answered is not None:
            logger.info(f"Generate {args.questions_answered} questions for each chunk")
            transformations.append(
                QuestionsAnsweredExtractor(questions=args.questions_answered, llm=llm)
            )

        pipeline = IngestionPipeline(transformations=transformations)
        # jww (2025-05-03): Make num_workers here customizable
        nodes = await pipeline.arun(documents=documents, num_workers=4)

        return TransformedNodesEvent(nodes=nodes)

    @step
    async def populate_vector_store(
        self,
        ev: TransformedNodesEvent | HaveVectorStoreEvent | HaveEmbeddingEvent,
        ctx: Context,
    ) -> HaveVectorIndexEvent | None:
        data = ctx.collect_events(
            ev, [TransformedNodesEvent, HaveVectorStoreEvent, HaveEmbeddingEvent]
        )
        if data is None:
            return None  # Not all required events have arrived yet
        nodes_event, vector_store_event, embedding_event = data
        nodes_event = cast(TransformedNodesEvent, nodes_event)
        vector_store_event = cast(HaveVectorStoreEvent, vector_store_event)
        embedding_event = cast(HaveEmbeddingEvent, embedding_event)

        args: Args = await ctx.get("args")
        nodes: Sequence[BaseNode] = nodes_event.nodes
        vector_store: BasePydanticVectorStore = vector_store_event.vector_store
        embed_model: BaseEmbedding = embedding_event.embed_model

        logger.info("Create storage context")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        vector_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=args.verbose,
        )
        return HaveVectorIndexEvent(vector_index=vector_index)

    @step
    async def persist_vector_index(
        self,
        ev: HaveVectorIndexEvent | HavePersistDirEvent | InMemoryVectorStoreEvent,
        ctx: Context,
    ) -> None:
        data = ctx.collect_events(
            ev, [HaveVectorIndexEvent, HavePersistDirEvent, InMemoryVectorStoreEvent]
        )
        if data is None:
            return None  # Not all required events have arrived yet
        vector_index_event, persist_dir_event, _ = data
        vector_index_event = cast(HaveVectorIndexEvent, vector_index_event)
        persist_dir_event = cast(HavePersistDirEvent, persist_dir_event)

        # Whenever we consume the HaveVectorIndexEvent, we must generate it
        # again so that other steps can also use it
        ctx.send_event(vector_index_event)

        vector_index: BaseIndex[IndexDict] = vector_index_event.vector_index
        persist_dir: Path = persist_dir_event.persist_dir

        logger.info("Write index to cache")
        vector_index.storage_context.persist(  # pyright: ignore[reportUnknownMemberType]
            persist_dir=persist_dir
        )

        return None

    @step
    async def retrieve_nodes(
        self,
        ev: HaveVectorIndexEvent | SearchInputEvent,
        ctx: Context,
    ) -> SearchResultEvent | None:
        data = ctx.collect_events(ev, [HaveVectorIndexEvent, SearchInputEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        vector_index_event, search_input_event = data
        vector_index_event = cast(HaveVectorIndexEvent, vector_index_event)
        search_input_event = cast(SearchInputEvent, search_input_event)

        # Whenever we consume the HaveVectorIndexEvent, we must generate it
        # again so that other steps can also use it
        ctx.send_event(vector_index_event)

        args: Args = await ctx.get("args")
        logger.info("Create retriever object")
        vector_index = vector_index_event.vector_index
        retriever = vector_index.as_retriever(similarity_top_k=args.top_k)
        logger.info("Retrieve nodes from vector index")
        nodes = await retriever.aretrieve(search_input_event.text)
        logger.info(f"{len(nodes)} nodes found in vector index")
        nodes = [{"text": node.text, "metadata": node.metadata} for node in nodes]
        return SearchResultEvent(nodes=nodes)

    @step
    async def command_search(
        self, ev: SearchResultEvent | HaveSearchCommand, ctx: Context
    ) -> StopEvent | None:
        data = ctx.collect_events(ev, [SearchResultEvent, HaveSearchCommand])
        if data is None:
            return None  # Not all required events have arrived yet
        nodes_event, _ = data
        nodes_event = cast(SearchResultEvent, nodes_event)

        nodes = nodes_event.nodes
        return StopEvent(json.dumps(nodes, indent=2))

    @step
    async def no_vector_index(
        self,
        ev: NoInputFilesEvent | InMemoryVectorStoreEvent,
        ctx: Context,
    ) -> NoVectorIndexEvent | None:
        data = ctx.collect_events(ev, [NoInputFilesEvent, InMemoryVectorStoreEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        return NoVectorIndexEvent()

    @step
    async def command_search_not_possible(
        self, ev: NoVectorIndexEvent | HaveSearchCommand, ctx: Context
    ) -> StopEvent | None:
        data = ctx.collect_events(ev, [NoVectorIndexEvent, HaveSearchCommand])
        if data is None:
            return None  # Not all required events have arrived yet
        no_vector_index_event, search_command = data
        no_vector_index_event = cast(NoVectorIndexEvent, no_vector_index_event)
        search_command = cast(HaveSearchCommand, search_command)

        # Whenever we consume the NoVectorIndexEvent, we must generate it
        # again so that other steps can also use it
        ctx.send_event(no_vector_index_event)

        # jww (2025-05-03): Why is this output twice?
        logger.error("Cannot search without a vector index")
        return StopEvent()

    @step
    async def query_only(
        self,
        ev: NoVectorIndexEvent | QueryInputEvent,
        ctx: Context,
    ) -> QueryBuiltEvent | None:
        data = ctx.collect_events(ev, [NoVectorIndexEvent, QueryInputEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        no_vector_index_event, query_input_event = data
        no_vector_index_event = cast(NoVectorIndexEvent, no_vector_index_event)
        query_input_event = cast(QueryInputEvent, query_input_event)

        # Whenever we consume the NoVectorIndexEvent, we must generate it
        # again so that other steps can also use it
        ctx.send_event(no_vector_index_event)

        logger.info("Build prompt")
        qa_prompt = PromptTemplate("QUERY: {query_str}\nANSWER: ")

        logger.info("Build query string")
        query = qa_prompt.format(
            query_str=query_input_event.query,
        )

        return QueryBuiltEvent(
            user_string=query_input_event.query,
            query=query,
            history=query_input_event.history,
        )

    @step
    async def query_with_context(
        self,
        ev: SearchResultEvent | QueryInputEvent,
        ctx: Context,
    ) -> QueryBuiltEvent | None:
        data = ctx.collect_events(ev, [SearchResultEvent, QueryInputEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        nodes_event, query_input_event = data
        nodes_event = cast(SearchResultEvent, nodes_event)
        query_input_event = cast(QueryInputEvent, query_input_event)

        context_nodes = nodes_event.nodes
        context_str = "\n".join([n["text"] for n in context_nodes])

        if len(query_input_event.history) > 0:
            chat_history_str = "\n".join(query_input_event.history)
            chat_history_str = "CHAT HISTORY: " + chat_history_str
        else:
            chat_history_str = ""

        logger.info("Build LLM prompt")
        qa_prompt = PromptTemplate(
            """
            CONTEXT information is below:\n
            ---------------------\n
            {context_str}\n\n
            {chat_history_str}\n
            ---------------------\n
            Given the context information and not prior knowledge,
            answer the query.\n
            QUERY: {query_str}\n
            ANSWER:
            """
        )

        logger.info("Build query string")
        query = qa_prompt.format(
            context_str=context_str,
            chat_history_str=chat_history_str,
            query_str=query_input_event.query,
        )

        return QueryBuiltEvent(
            user_string=query_input_event.query,
            query=query,
            history=query_input_event.history,
        )

    @step
    async def query_llm(
        self,
        ev: QueryBuiltEvent | HaveLLMEvent,
        ctx: Context,
    ) -> QueryResponseEvent | None:
        data = ctx.collect_events(ev, [QueryBuiltEvent, HaveLLMEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        query_built_event, llm_event = data
        query_built_event = cast(QueryBuiltEvent, query_built_event)
        llm_event = cast(HaveLLMEvent, llm_event)

        # Whenever we consume the HaveLLMEvent, we must generate it again so
        # that other steps can also use it
        ctx.send_event(llm_event)

        args: Args = await ctx.get("args")

        logger.info("Submit query to LLM")
        query = query_built_event.query
        llm = llm_event.llm
        if args.streaming:
            generator = await llm.astream_complete(query)
            async for response in generator:
                msg = response.delta
                if msg is not None:
                    msg = clean_special_tokens(msg)
                    # Allow the workflow to stream this piece of response
                    ctx.write_event_to_stream(ProgressEvent(msg=msg))

            # jww (2025-05-03): What to do here?
            return QueryResponseEvent(response="", history=[])
        else:
            response = await llm.acomplete(query)
            response = clean_special_tokens(response.text)
            return QueryResponseEvent(
                response=response,
                history=query_built_event.history
                + [f"User: {query_built_event.user_string}", f"Assistant: {response}"],
            )

    @step
    async def command_query(
        self,
        ev: QueryResponseEvent | HaveQueryCommand,
        ctx: Context,
    ) -> StopEvent | None:
        data = ctx.collect_events(ev, [QueryResponseEvent, HaveQueryCommand])
        if data is None:
            return None  # Not all required events have arrived yet
        query_response_event, _ = data
        query_response_event = cast(QueryResponseEvent, query_response_event)

        response = query_response_event.response
        return StopEvent(response)

    @step
    async def command_chat(
        self,
        ev: QueryResponseEvent | HaveChatCommand,
        ctx: Context,
    ) -> CommandEvent | None:
        data = ctx.collect_events(ev, [QueryResponseEvent, HaveChatCommand])
        if data is None:
            return None  # Not all required events have arrived yet
        print(data)
        query_response_event, _ = data
        query_response_event = cast(QueryResponseEvent, query_response_event)

        response = query_response_event.response
        print(f"[LLM] {response}")

        return CommandEvent(history=query_response_event.history)


### Main


async def rag_client(args: Args):
    rag_workflow = RAGWorkflow(verbose=args.verbose, timeout=args.timeout)
    ctx = Context(rag_workflow)
    await ctx.set("args", args)

    handler = rag_workflow.run(ctx=ctx)

    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.msg, end="")

    result = await handler
    # draw_most_recent_execution(workflow, filename="execution_log.html")
    # draw_all_possible_flows(MyWorkflow, filename="streaming_workflow.html")
    return result


def main(args: Args):
    logging.basicConfig(
        stream=sys.stdout,
        encoding="utf-8",
        level=logging.INFO if args.verbose else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",  # Log message format
        datefmt="%H:%M:%S",
    )

    result = asyncio.run(rag_client(args))
    print(result)


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
