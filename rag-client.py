#!/usr/bin/env python

from abc import abstractmethod
import asyncio
import base64
import hashlib
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, NoReturn, cast, no_type_check, override

from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores.simple import SimpleVectorStore
from pydantic import RootModel
import typed_argparse as tap
from llama_index.core import (
    Document,
    PromptTemplate,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.indices import load_index_from_storage
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import (
    BaseNode,
    Document,
    Node,
    NodeWithScore,
)
from llama_index.core.storage.storage_context import StorageContext
# from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.postgres import PGVectorStore
from orgparse.node import OrgNode
from typed_argparse import TypedArgs, arg
from xdg_base_dirs import xdg_cache_home

os.environ["TOKENIZERS_PARALLELISM"] = "false"


### Utility functions


class NodeWithScoreList(RootModel[list[NodeWithScore]]):
    pass


def error(msg: str) -> NoReturn:
    print(msg, sys.stderr)
    sys.exit(1)


def parse_prefixes(prefixes: list[str], s: str) -> tuple[str | None, str]:
    for prefix in prefixes:
        if s.startswith(prefix):
            return prefix, s[len(prefix) :]
    return None, s  # No matching prefix found


def list_files(directory: Path, recursive=False) -> list[Path]:
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
    def construct(self, _verbose: bool = False) -> BasePydanticVectorStore:
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
    def construct(self, verbose: bool = False) -> BasePydanticVectorStore:
        if verbose:
            print("Setup PostgreSQL vector store")
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

    def construct(self, model: str, verbose: bool = False) -> BaseEmbedding | NoReturn:
        prefix, model = parse_prefixes(
            [
                "HuggingFace:",
                "Gemini:",
                "Ollama:",
                "OpenAILike:",
                "OpenAI:",
                "LlamaCpp:",
            ],
            model,
        )
        if verbose:
            print(f"Load embedding {prefix}:{model}")
        if prefix == "HuggingFace:":
            return HuggingFaceEmbedding(model_name=model)
        elif prefix == "Gemini:":
            return GeminiEmbedding(model_name=model)
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

        if verbose:
            print(f"Load LLM {prefix}:{model}")
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


class HaveSearchTextEvent(Event):
    text: str


class HaveSearchCommand(Event):
    pass


class HaveQueryCommand(Event):
    query: str


class HaveChatCommand(Event):
    pass


class HaveVectorStoreEvent(Event):
    vector_store: BasePydanticVectorStore


class PopulatedVectorStoreEvent(Event):
    vector_store: BasePydanticVectorStore


class HaveVectorIndexEvent(Event):
    vector_index: BaseIndex[IndexDict]


class NodesRetrievedEvent(Event):
    retrieved_nodes: list[dict[str, Any]]


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


class HaveNodesEvent(Event):
    nodes: list[BaseNode]


class TransformedNodesEvent(Event):
    nodes: list[BaseNode]


class EmbeddedNodesEvent(Event):
    nodes: list[BaseNode]


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
    async def rag_workflow(
        self, _ev: StartEvent, ctx: Context
    ) -> (
        HaveSearchTextEvent
        | HaveSearchCommand
        | HaveQueryCommand
        | HaveChatCommand
        | StopEvent
    ):
        args: Args = await ctx.get("args")
        match args.command:
            case "search":
                ctx.send_event(HaveSearchTextEvent(text=args.args[0]))
                return HaveSearchCommand()
            case "query":
                ctx.send_event(HaveSearchTextEvent(text=args.args[0]))
                return HaveQueryCommand(query=args.args[0])
            case "chat":
                return HaveChatCommand()
            case _:
                return StopEvent(f"Command unrecognized: {args.command}")

    @step
    async def load_vector_store(
        self, _ev: StartEvent, ctx: Context
    ) -> HaveVectorStoreEvent | InMemoryVectorStoreEvent | NoInMemoryVectorStoreEvent:
        args: Args = await ctx.get("args")
        if args.db_name is None:
            if args.verbose:
                print("In-memory vector store")
            ctx.send_event(InMemoryVectorStoreEvent())
            return HaveVectorStoreEvent(vector_store=SimpleVectorStore())
        else:
            if args.verbose:
                print("Postgres vector store")
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
            ).construct(args.verbose)
            return HaveVectorStoreEvent(vector_store=vector_store)

    @step
    async def load_embedding(
        self, _ev: StartEvent, ctx: Context
    ) -> HaveEmbeddingEvent | None:
        args: Args = await ctx.get("args")
        if args.embed_model is None:
            if args.verbose:
                print("No embedding model")
            return None
        else:
            embed_model = EmbeddingObject(base_url=args.embed_base_url).construct(
                model=args.embed_model, verbose=args.verbose
            )
            return HaveEmbeddingEvent(embed_model=embed_model)

    @step
    async def load_llm(
        self, _ev: StartEvent, ctx: Context
    ) -> HaveLLMEvent | NoLLMEvent:
        args: Args = await ctx.get("args")
        if args.llm is None:
            if args.verbose:
                print("No LLM")
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
            if args.verbose:
                print("No input files")
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
        args: Args = await ctx.get("args")
        fingerprint: str = ev.fingerprint
        persist_dir = cache_dir(fingerprint)
        if args.verbose:
            print(f"Cache directory = {persist_dir}")
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

        args: Args = await ctx.get("args")
        if args.verbose:
            print("Load index from cache")
        global Settings
        embed_model: BaseEmbedding = embedding_event.embed_model
        Settings.embed_model = embed_model
        if args.chunk_size is not None:
            Settings.chunk_size = args.chunk_size
        if args.chunk_overlap is not None:
            Settings.chunk_overlap = args.chunk_overlap
        persist_dir: Path = persist_dir_event.persist_dir
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(persist_dir))
        )
        return HaveVectorIndexEvent(vector_index=vector_index)

    @step
    async def load_index_from_vector_store(
        self,
        ev: HaveVectorStoreEvent | NoInputFilesEvent | NoInMemoryVectorStoreEvent,
        ctx: Context,
    ) -> HaveVectorIndexEvent | None:
        data = ctx.collect_events(
            ev, [HaveVectorStoreEvent, NoInputFilesEvent, NoInMemoryVectorStoreEvent]
        )
        if data is None:
            return None  # Not all required events have arrived yet
        vector_store_event, _, _ = data

        vector_store = vector_store_event.vector_store

        args: Args = await ctx.get("args")
        if args.verbose:
            print("Load index from database")
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            # embed_model=model,        # jww (2025-05-02): Is this needed?
            show_progress=args.verbose,
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

        args: Args = await ctx.get("args")

        file_extractor: dict[str, BaseReader] = {".org": OrgReader()}

        if args.verbose:
            print("Read documents from disk")
        input_files: list[Path] = input_files_event.input_files
        documents = SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=file_extractor,
        ).load_data()
        return HaveDocumentsEvent(documents=documents)

    @step
    async def split_documents(
        self, ev: HaveDocumentsEvent, ctx: Context
    ) -> HaveNodesEvent:
        args: Args = await ctx.get("args")
        documents: list[Document] = ev.documents

        # Basic text chunking
        splitter = SentenceSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            include_metadata=True,
        )

        # # Semantic-aware splitting
        # semantic_splitter = SemanticSplitterNodeParser(
        #     buffer_size=256,
        #     breakpoint_percentile_threshold=95
        # )

        nodes: list[BaseNode] = splitter(documents)
        return HaveNodesEvent(nodes=nodes)

    @step
    async def transform_nodes(
        self, ev: HaveNodesEvent | NoLLMEvent, ctx: Context
    ) -> TransformedNodesEvent | None:
        data = ctx.collect_events(ev, [HaveNodesEvent, NoLLMEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        nodes_event, _ = data

        # args: Args = await ctx.get("args")
        nodes: list[BaseNode] = nodes_event.nodes
        return TransformedNodesEvent(nodes=nodes)

    @step
    async def transform_nodes_with_llm(
        self, ev: HaveNodesEvent | HaveLLMEvent, ctx: Context
    ) -> TransformedNodesEvent | None:
        data = ctx.collect_events(ev, [HaveNodesEvent, HaveLLMEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        nodes_event, llm_event = data

        args: Args = await ctx.get("args")
        nodes: list[BaseNode] = nodes_event.nodes
        if args.questions_answered is not None:
            if args.verbose:
                print(f"Generate {args.questions_answered} questions for each chunk")
            llm: LLM = llm_event.llm
            questions_answered_extractor = QuestionsAnsweredExtractor(
                questions=args.questions_answered, llm=llm
            )
            nodes = await questions_answered_extractor.acall(nodes)
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
        nodes: list[BaseNode] = nodes_event.nodes
        vector_store: BasePydanticVectorStore = vector_store_event.vector_store
        embed_model: BaseEmbedding = embedding_event.embed_model

        if args.verbose:
            print("Create storage context")
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

        args: Args = await ctx.get("args")

        vector_index: BaseIndex[IndexDict] = vector_index_event.vector_index
        persist_dir: Path = persist_dir_event.persist_dir

        if args.verbose:
            print("Write index to cache")
        vector_index.storage_context.persist(persist_dir=persist_dir)

        return None

    @step
    async def retrieve_nodes(
        self,
        ev: HaveVectorIndexEvent | HaveSearchTextEvent,
        ctx: Context,
    ) -> NodesRetrievedEvent | None:
        data = ctx.collect_events(ev, [HaveVectorIndexEvent, HaveSearchTextEvent])
        if data is None:
            return None  # Not all required events have arrived yet
        vector_index_event, search_text_event = data
        vector_index_event = cast(HaveVectorIndexEvent, vector_index_event)
        search_text_event = cast(HaveSearchTextEvent, search_text_event)

        args: Args = await ctx.get("args")
        if args.verbose:
            print("Create retriever object")
        vector_index = vector_index_event.vector_index
        retriever = vector_index.as_retriever(similarity_top_k=args.top_k)
        if args.verbose:
            print("Retrieve nodes from vector index")
        nodes = await retriever.aretrieve(search_text_event.text)
        if args.verbose:
            print(f"{len(nodes)} nodes found in vector index")
        nodes = [{"text": node.text, "metadata": node.metadata} for node in nodes]
        return NodesRetrievedEvent(retrieved_nodes=nodes)

    @step
    async def search_vector_index(
        self, ev: NodesRetrievedEvent | HaveSearchCommand, ctx: Context
    ) -> StopEvent | None:
        data = ctx.collect_events(ev, [NodesRetrievedEvent, HaveSearchCommand])
        if data is None:
            return None  # Not all required events have arrived yet
        nodes_event, _ = data
        nodes_event = cast(NodesRetrievedEvent, nodes_event)

        nodes = nodes_event.retrieved_nodes
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
    async def query_llm(
        self,
        ev: NoVectorIndexEvent | HaveLLMEvent | HaveQueryCommand,
        ctx: Context,
    ) -> StopEvent | None:
        data = ctx.collect_events(
            ev, [NoVectorIndexEvent, HaveLLMEvent, HaveQueryCommand]
        )
        if data is None:
            return None  # Not all required events have arrived yet
        _, llm_event, query_command = data
        llm_event = cast(HaveLLMEvent, llm_event)
        query_command = cast(HaveQueryCommand, query_command)

        args: Args = await ctx.get("args")
        if args.verbose:
            print("Create retriever object")

        if args.verbose:
            print("Build LLM prompt")
        qa_prompt = PromptTemplate("QUERY: {query_str}\nANSWER: ")

        if args.verbose:
            print("Build query string")
        query = qa_prompt.format(
            query_str=query_command.query,
        )
        if args.verbose:
            print("Submit query to LLM")
        llm = llm_event.llm
        response = await llm.acomplete(query)
        response = clean_special_tokens(response.text)

        return StopEvent(response)

    @step
    async def query_vector_index(
        self,
        ev: NodesRetrievedEvent | HaveLLMEvent | HaveQueryCommand,
        ctx: Context,
    ) -> StopEvent | None:
        data = ctx.collect_events(
            ev, [NodesRetrievedEvent, HaveLLMEvent, HaveQueryCommand]
        )
        if data is None:
            return None  # Not all required events have arrived yet
        nodes_event, llm_event, query_command = data
        nodes_event = cast(NodesRetrievedEvent, nodes_event)
        llm_event = cast(HaveLLMEvent, llm_event)
        query_command = cast(HaveQueryCommand, query_command)

        args: Args = await ctx.get("args")
        if args.verbose:
            print("Create retriever object")

        context_nodes = nodes_event.retrieved_nodes
        context_str = "\n".join([n["text"] for n in context_nodes])

        # if event.mode == "chat":
        #     chat_history_str = (
        #         "\n".join([f"{msg.role}: {msg.content}" for msg in event.chat_history])
        #         if event.chat_history
        #         else ""
        #     )
        #     chat_history_str = "CHAT HISTORY: " + chat_history_str
        # else:
        #     chat_history_str = ""

        if args.verbose:
            print("Build LLM prompt")
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

        if args.verbose:
            print("Build query string")
        query = qa_prompt.format(
            context_str=context_str,
            # chat_history_str=chat_history_str,
            query_str=query_command.query,
        )
        if args.verbose:
            print("Submit query to LLM")
        llm = llm_event.llm
        response = await llm.acomplete(query)
        response = clean_special_tokens(response.text)

        return StopEvent(response)

    @step
    async def chat_vector_index(
        self,
        ev: HaveVectorIndexEvent | HaveChatCommand,
        ctx: Context,
    ) -> StopEvent | None:
        data = ctx.collect_events(ev, [HaveVectorIndexEvent, HaveChatCommand])
        if data is None:
            return None  # Not all required events have arrived yet
        vector_index_event, chat_command = data

        args: Args = await ctx.get("args")
        if args.verbose:
            print("Create retriever object")
        return StopEvent()


### Main


async def rag_client(args: Args):
    rag_workflow = RAGWorkflow(verbose=args.verbose, timeout=args.timeout)
    ctx = Context(rag_workflow)
    await ctx.set("args", args)
    return await rag_workflow.run(ctx=ctx)


def main(args: Args):
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
