#!/usr/bin/env python

import asyncio
import base64
import hashlib
import json
import logging
import os
import sys
from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal, NoReturn, no_type_check, override

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.storage.chat_store import SimpleChatStore
import typed_argparse as tap
from llama_index.core import (
    ChatPromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.indices import (
    load_index_from_storage,  # pyright: ignore[reportUnknownVariableType]
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import BaseNode, Document, Node, TransformComponent
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


### Workflows


@dataclass
class RAGWorkflow:
    args: Args
    vector_store: BasePydanticVectorStore = field(default=SimpleVectorStore())
    embed_model: BaseEmbedding | None = None
    retriever: BaseRetriever | None = None
    llm: LLM | None = None
    fingerprint: str = ""
    chat_memory: ChatMemoryBuffer | ChatSummaryMemoryBuffer | None = None
    chat_history: list[ChatMessage] = []
    chat_engine: ContextChatEngine | None = None
    vector_index: BaseIndex[IndexDict] | None = None

    async def initialize(self):
        self.vector_store, self.embed_model, self.llm = await asyncio.gather(
            self.load_vector_store(),
            self.load_embedding(),
            self.load_llm(),
        )

        input_files = await self.find_input_files()
        if input_files is None:
            self.vector_index = await self.load_index_from_vector_store()
        else:
            fp: str = await self.determine_fingerprint(input_files)
            persist_dir = cache_dir(fp)
            if self.fingerprint == fp and os.path.isdir(persist_dir):
                self.vector_index = await self.load_index_from_cache(persist_dir)
            else:
                documents = await self.read_documents(input_files)
                nodes = await self.split_documents(documents)
                self.vector_index = await self.populate_vector_store(nodes)

                if isinstance(self.vector_store, SimpleVectorStore):
                    logger.info("Write index to cache")
                    self.vector_index.storage_context.persist(  # pyright: ignore[reportUnknownMemberType]
                        persist_dir=persist_dir
                    )

        logger.info("Create retriever object")
        args: Args = self.args
        self.retriever = self.vector_index.as_retriever(similarity_top_k=args.top_k)

        # jww (2025-05-04): TODO
        query = ""
        if self.llm is None:
            nodes = await self.retrieve_nodes(query)
        else:
            chat_store = SimpleChatStore()
            # chat_store = PostgresChatStore.from_uri(
            #     uri="postgresql+asyncpg://postgres:password@127.0.0.1:5432/database",
            # )

            # jww (2025-05-04): Make this configurable
            self.chat_memory = ChatMemoryBuffer.from_defaults(  # pyright: ignore[reportUnknownMemberType]
                token_limit=1500,
                chat_store=chat_store,
                chat_story_key="user1",
                llm=self.llm,
            )
            self.chat_memory = ChatSummaryMemoryBuffer.from_defaults(  # pyright: ignore[reportUnknownMemberType]
                token_limit=1500,
                summarize_prompt=(
                    "The following is a conversation between the user and assistant. "
                    "Write a concise summary about the contents of this conversation."
                ),
                chat_store=chat_store,
                chat_story_key="user1",
                llm=self.llm,
            )
            self.chat_engine = ContextChatEngine(
                retriever=self.retriever,
                llm=self.llm,
                memory=self.chat_memory,
                prefix_messages=[
                    ChatMessage(
                        # jww (2025-05-03): This should be configurable
                        content="You are a helpful AI assistant.",
                        role=MessageRole.SYSTEM,
                    ),
                ],
            )

            new_message = ChatMessage(role="user", content=query)
            generator = await self.chat_engine.astream_chat(  # pyright: ignore[reportUnknownMemberType]
                new_message, self.chat_history
            )
            token: str
            async for token in generator.async_response_gen:
                print(token, end="")

    async def execute(self, _command: str, _args: list[str]) -> str:
        return ""

    #     match command:
    #         case "search":
    #             return SearchInputEvent(text=args.args[0])
    #         case "query":
    #             ctx.send_event(SearchInputEvent(text=args.args[0]))
    #             return QueryInputEvent(query=args.args[0], history=history)
    #         case "chat":
    #             print("[USER]")
    #             query = input("> ")
    #             ctx.send_event(SearchInputEvent(text=query))
    #             return QueryInputEvent(query=query, history=history)
    #         case _:
    #             return StopEvent(f"Command unrecognized: {args.command}")

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
        fingerprint = [
            collection_hash(input_files),
            hashlib.sha512(repr(self.args).encode("utf-8")).hexdigest(),
        ]
        final_hash = "\n".join(fingerprint).encode("utf-8")
        final_base64 = base64.b64encode(final_hash).decode("utf-8")
        return final_base64[0:32]

    async def load_index_from_cache(self, persist_dir: Path) -> BaseIndex[IndexDict]:
        args: Args = self.args
        logger.info("Load index from cache")
        global Settings
        Settings.embed_model = self.embed_model
        Settings.chunk_size = args.chunk_size
        Settings.chunk_overlap = args.chunk_overlap
        return load_index_from_storage(  # pyright: ignore[reportUnknownVariableType]
            StorageContext.from_defaults(persist_dir=str(persist_dir))
        )

    async def load_index_from_vector_store(self) -> VectorStoreIndex:
        args: Args = self.args
        logger.info("Load index from database")
        return VectorStoreIndex.from_vector_store(  # pyright: ignore[reportUnknownMemberType]
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            show_progress=args.verbose,
        )

    async def read_documents(self, input_files: list[Path]) -> list[Document]:
        file_extractor: dict[str, BaseReader] = {".org": OrgReader()}

        logger.info("Read documents from disk")
        return SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=file_extractor,
        ).load_data()

    async def split_documents(self, documents: list[Document]) -> Sequence[BaseNode]:
        args: Args = self.args

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
    ) -> VectorStoreIndex:
        args: Args = self.args

        logger.info("Create storage context")
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        return VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=args.verbose,
        )

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

    async def build_query(
        self,
        query: str,
        nodes: list[dict[str, Any]],  # pyright: ignore[reportExplicitAny]
    ) -> list[ChatMessage]:
        message_templates: list[ChatMessage] = [
            # jww (2025-05-03): This should be configurable
            ChatMessage(
                content="You are a helpful AI assistant.",
                role=MessageRole.SYSTEM,
            ),
        ]

        if len(nodes) > 0:
            context_str = "\n".join([n["text"] for n in nodes])
            message_templates.append(
                ChatMessage(
                    content="""
                Context information is below:
                ---------------------
                {context_str}
                ---------------------
                """,
                    role=MessageRole.SYSTEM,
                ),
            )
        else:
            context_str = ""

        logger.info("Build LLM prompt")
        message_templates.append(
            ChatMessage(
                content="""
                Given the context information and not prior knowledge,
                answer the query: {query_str}
                """,
                role=MessageRole.USER,
            ),
        )
        chat_template = ChatPromptTemplate(message_templates=message_templates)

        logger.info("Build query string")
        return chat_template.format_messages(
            context_str=context_str,
            query_str=query,
        )

    # async def query_llm(self, list[ChatMessage]) -> None:
    #     args: Args = self.args

    #     logger.info("Submit query to LLM")
    #     query = query_built_event.query
    #     llm = llm_event.llm
    #     if args.streaming:
    #         generator = await llm.astream_complete(query)
    #         async for response in generator:
    #             msg = response.delta
    #             if msg is not None:
    #                 msg = clean_special_tokens(msg)
    #                 # Allow the workflow to stream this piece of response
    #                 ctx.write_event_to_stream(ProgressEvent(msg=msg))

    #         # jww (2025-05-03): What to do here?
    #         return QueryResponseEvent(response="", history=[])
    #     else:
    #         response = await llm.acomplete(query)
    #         response = clean_special_tokens(response.text)
    #         return QueryResponseEvent(
    #             response=response,
    #             history=query_built_event.history
    #             + [f"User: {query_built_event.user_string}", f"Assistant: {response}"],
    #         )


### Main


async def rag_client(args: Args) -> str:
    rag_workflow = RAGWorkflow(args)

    await rag_workflow.initialize()
    return await rag_workflow.execute(args.command, args.args)


def main(args: Args):
    logging.basicConfig(
        stream=sys.stdout,
        encoding="utf-8",
        level=logging.INFO if args.verbose else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",  # Log message format
        datefmt="%H:%M:%S",
    )
    print(asyncio.run(rag_client(args)))


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
