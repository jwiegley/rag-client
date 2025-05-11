# pyright: reportMissingTypeStubs=false
# pyright: reportExplicitAny=false

import asyncio
import base64
import hashlib
import logging
import os
import sys
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from dataclass_wizard import YAMLWizard
from functools import cache
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    NoReturn,
    TypeVar,
    cast,
    final,
    no_type_check,
    override,
)
from urllib.parse import urlparse

import numpy as np
import psycopg2
from llama_index.core import (
    load_indices_from_storage,  # pyright: ignore[reportUnknownVariableType]
)
from llama_index.core import (
    KeywordTableIndex,
    QueryBundle,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.evaluation import GuidelineEvaluator, RelevancyEvaluator
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
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
    CodeSplitter,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
)
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    RetryQueryEngine,
    RetrySourceQueryEngine,
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import (
    BaseNode,
    Document,
    Node,
    NodeWithScore,
    TransformComponent,
)
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.storage_context import DEFAULT_PERSIST_DIR
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.indices.managed.bge_m3 import BGEM3Index
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.perplexity import Perplexity

# from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.index_store.postgres import PostgresIndexStore
from llama_index.vector_stores.postgres import PGVectorStore
from orgparse.node import OrgNode
from typed_argparse import TypedArgs
from xdg_base_dirs import xdg_cache_home

IndexList = list[BaseIndex[IndexDict]]

# Utility functions


logger = logging.getLogger("rag")


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


async def list_files(directory: Path, recursive: bool = False) -> list[Path]:
    if recursive:
        # Run the blocking os.walk in a thread
        def walk_files() -> list[Path]:
            file_list: list[Path] = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file not in [".", ".."]:
                        file_list.append(Path(root) / Path(file))
            return file_list

        return await asyncio.to_thread(walk_files)
    else:
        # Run blocking I/O in threads
        files = await asyncio.to_thread(os.listdir, directory)
        return [
            directory / f
            for f in files
            if await asyncio.to_thread(
                os.path.isfile,
                os.path.join(directory, f),
            )
        ]


async def read_files(
    read_from: str,
    recursive: bool = False,
) -> list[Path] | NoReturn:
    if read_from == "-":
        # Reading from stdin is still blocking; consider using asyncio streams
        # if needed
        input_files = [Path(line.strip()) for line in sys.stdin if line.strip()]
        if not input_files:
            error("No filenames provided on standard input")
        return input_files
    elif await asyncio.to_thread(os.path.isdir, read_from):
        return await list_files(Path(read_from), recursive)
    elif await asyncio.to_thread(os.path.isfile, read_from):
        return [Path(read_from)]
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


# Args


@dataclass
class LLMConfig(YAMLWizard):
    provider: str
    model: str
    base_url: str | None = None
    api_key: str = "fake_key"
    api_version: str = ""


@dataclass
class Config(YAMLWizard):
    embedding: LLMConfig | None
    db_conn: str | None = None
    query_instruction: str | None = None
    semantic_splitter_embed_provider: str | None = None
    semantic_splitter_embed_model: str | None = None
    semantic_splitter_embed_api_key: str | None = None
    semantic_splitter_embed_api_version: str | None = None
    semantic_splitter_embed_base_url: str | None = None
    semantic_splitter_query_instruction: str | None = None
    questions_answered: int | None = None
    questions_answered_provider: str | None = None
    questions_answered_model: str | None = None
    questions_answered_api_key: str | None = None
    questions_answered_api_version: str | None = None
    questions_answered_base_url: str | None = None
    hybrid_search: bool = False
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: str = "fake_key"
    llm_api_version: str | None = None
    llm_base_url: str | None = None
    chat_user: str | None = None
    metadata_extractor_provider: str | None = None
    metadata_extractor_model: str | None = None
    metadata_extractor_api_key: str | None = None
    metadata_extractor_api_version: str | None = None
    metadata_extractor_base_url: str | None = None
    collect_keywords: bool = False
    keywords_provider: str | None = None
    keywords_model: str | None = None
    keywords_api_key: str | None = None
    keywords_api_version: str | None = None
    keywords_base_url: str | None = None
    retries: bool = False
    source_retries: bool = False
    evaluator_provider: str | None = None
    evaluator_model: str | None = None
    evaluator_api_key: str | None = None
    evaluator_api_version: str | None = None
    evaluator_base_url: str | None = None
    summarize_chat: bool = False
    num_workers: int = DEFAULT_NUM_WORKERS
    streaming: bool = False
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    hnsw_dist_method: str = "vector_cosine_ops"
    embed_dim: int = 512
    chunk_size: int = 512
    chunk_overlap: int = 20
    splitter: str = "Sentence"
    code_language: str = "python"
    code_chunk_lines: int = 40
    code_chunk_lines_overlap: int = 15
    code_max_chars: int = 1500
    buffer_size: int = 256
    breakpoint_percentile_threshold: int = 95
    window_size: int = 3
    top_k: int = 3
    timeout: int = 60
    temperature: float = 1.0
    max_tokens: int = 200
    context_window: int = 2048
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    gpu_layers: int = -1
    token_limit: int = 1500
    recursive: bool = False
    host: str = "localhost"
    port: int = 8000
    reload_server: bool = False


class Args(TypedArgs):
    from_: str | None
    verbose: bool
    config: str
    command: str
    args: list[str]


# Readers


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
        text = "\n".join(
            get_text_from_org_node(
                node,
                format=self.text_formatting,
            )
        )
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


# Embeddings


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


# Retrievers


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


# Workflows


T = TypeVar("T")


class PostgresDetails(Generic[T]):
    connection_string: str
    database: str
    host: str
    password: str
    port: int
    user: str

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

        parsed = urlparse(self.connection_string)
        self.database = parsed.path.lstrip("/")
        self.host = parsed.hostname or "localhost"
        self.password = parsed.password or ""
        self.port = parsed.port or 5432
        self.user = parsed.username or "postgres"

    def unpickle_from_table(self, tablename: str, row_id: int) -> T | NoReturn:
        import pickle

        import psycopg2

        # Connect to PostgreSQL
        with psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT data FROM {tablename} WHERE id = %s",
                    (row_id,),
                )
                row = cur.fetchone()
                if row is None:
                    error(f"Data not in table {tablename} row {row_id}")

                binary_data = row[0]  # pyright: ignore[reportAny]
                if isinstance(binary_data, memoryview):
                    binary_data = binary_data.tobytes()
                return cast(T, pickle.loads(binary_data))

    def pickle_to_table(self, tablename: str, row_id: int, data: T):
        import pickle

        import psycopg2

        # Connect to PostgreSQL
        with psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {tablename} (
                        id SERIAL PRIMARY KEY,
                        data BYTEA
                    )
                """
                )

                pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

                cur.execute(
                    f"""
                    INSERT INTO {tablename} (id, data)
                    VALUES (%s, %s)
                    ON CONFLICT (id)
                    DO UPDATE SET data = EXCLUDED.data
                """,
                    (row_id, psycopg2.Binary(pickled)),
                )


MultiEmbedStore = dict[
    Literal["dense_vecs", "lexical_weights", "colbert_vecs"],
    np.ndarray[Any, Any] | list[dict[str, float]] | list[np.ndarray[Any, Any]],
]


class BGEM3Embedding:
    @classmethod
    def load_from_postgres(
        cls,
        uri: str,
        storage_context: StorageContext,
        weights_for_different_modes: list[float],
        model_name: str = "BAAI/bge-m3",
        index_name: str = "",
    ) -> BGEM3Index:
        index = BGEM3Index(
            model_name=model_name,
            index_name=index_name,
            index_struct=storage_context.index_store.index_structs()[
                0
            ],  # pyright: ignore[reportArgumentType]
            storage_context=storage_context,
            weights_for_different_modes=weights_for_different_modes,
        )
        details: PostgresDetails[MultiEmbedStore] = PostgresDetails(uri)
        docs_pos_to_node_id = {
            int(k): v for k, v in index.index_struct.nodes_dict.items()
        }
        index._docs_pos_to_node_id = (  # pyright: ignore[reportPrivateUsage]
            docs_pos_to_node_id
        )
        index._multi_embed_store = (  # pyright: ignore[reportPrivateUsage]
            details.unpickle_from_table(
                "multi_embed_store",
                1,
            )
        )
        return index

    @classmethod
    def persist_to_postgres(cls, uri: str, index: BGEM3Index):
        details: PostgresDetails[MultiEmbedStore] = PostgresDetails(uri)
        details.pickle_to_table(
            "multi_embed_store",
            1,
            index._multi_embed_store,  # pyright: ignore[reportArgumentType, reportUnknownMemberType, reportPrivateUsage]
        )


@dataclass
class Models:
    embed_llm: BaseEmbedding | BGEM3Embedding | None = None
    semantic_splitter_embed_llm: BaseEmbedding | BGEM3Embedding | None = None
    llm: LLM | None = None
    questions_answered_llm: LLM | None = None
    metadata_extractor_llm: LLM | None = None
    keywords_llm: LLM | None = None
    evaluator_llm: LLM | None = None


@dataclass
class RAGWorkflow:
    config: Config
    fingerprint: str | None = None
    chat_memory: ChatMemoryBuffer | ChatSummaryMemoryBuffer | None = None
    chat_history: list[ChatMessage] | None = None
    chat_engine: BaseChatEngine | None = None

    async def initialize(self, verbose: bool = False) -> Models:
        embed_llm = (
            self.load_embedding(
                self.config.embedding.provider,
                self.config.embedding.model,
                self.config.timeout,
                self.config.embedding.api_key,
                self.config.embedding.api_version,
                self.config.embedding.base_url,
                self.config.query_instruction,
                self.config.num_workers,
                verbose=verbose,
            )
            if self.config.embedding
            else awaitable_none()
        )
        semantic_splitter_embed_llm = (
            self.load_embedding(
                self.config.semantic_splitter_embed_provider,
                self.config.semantic_splitter_embed_model,
                self.config.timeout,
                self.config.semantic_splitter_embed_api_key,
                self.config.semantic_splitter_embed_api_version,
                self.config.semantic_splitter_embed_base_url,
                self.config.semantic_splitter_query_instruction,
                self.config.num_workers,
                verbose=verbose,
            )
            if self.config.semantic_splitter_embed_provider
            and self.config.semantic_splitter_embed_model
            else awaitable_none()
        )
        llm = (
            self.load_llm(
                self.config.llm_provider,
                self.config.llm_model,
                self.config.timeout,
                self.config.llm_api_key,
                self.config.llm_api_version,
                self.config.llm_base_url,
                verbose=verbose,
            )
            if self.config.llm_provider and self.config.llm_model
            else awaitable_none()
        )
        questions_answered_llm = (
            self.load_llm(
                self.config.questions_answered_provider,
                self.config.questions_answered_model,
                self.config.timeout,
                self.config.questions_answered_api_key,
                self.config.questions_answered_api_version,
                self.config.questions_answered_base_url,
                verbose=verbose,
            )
            if self.config.questions_answered_provider
            and self.config.questions_answered_model
            else awaitable_none()
        )
        metadata_extractor_llm = (
            self.load_llm(
                self.config.metadata_extractor_provider,
                self.config.metadata_extractor_model,
                self.config.timeout,
                self.config.metadata_extractor_api_key,
                self.config.metadata_extractor_api_version,
                self.config.metadata_extractor_base_url,
                verbose=verbose,
            )
            if self.config.metadata_extractor_provider
            and self.config.metadata_extractor_model
            else awaitable_none()
        )
        keywords_llm = (
            self.load_llm(
                self.config.keywords_provider,
                self.config.keywords_model,
                self.config.timeout,
                self.config.keywords_api_key,
                self.config.keywords_api_version,
                self.config.keywords_base_url,
                verbose=verbose,
            )
            if self.config.keywords_provider and self.config.keywords_model
            else awaitable_none()
        )
        evaluator_llm = (
            self.load_llm(
                self.config.evaluator_provider,
                self.config.evaluator_model,
                self.config.timeout,
                self.config.evaluator_api_key,
                self.config.evaluator_api_version,
                self.config.evaluator_base_url,
                verbose=verbose,
            )
            if self.config.evaluator_provider and self.config.evaluator_model
            else awaitable_none()
        )

        return Models(
            embed_llm=await embed_llm,
            semantic_splitter_embed_llm=await semantic_splitter_embed_llm,
            llm=await llm,
            questions_answered_llm=await questions_answered_llm,
            metadata_extractor_llm=await metadata_extractor_llm,
            keywords_llm=await keywords_llm,
            evaluator_llm=await evaluator_llm,
        )

    @classmethod
    async def load_config(cls, config: Path) -> Config:
        if os.path.isfile(config):
            cfg = Config.from_yaml_file(  # pyright: ignore[reportUnknownMemberType]
                str(config)
            )
            if isinstance(cfg, Config):
                if cfg.embedding is not None:
                    if cfg.embedding.provider == "BGEM3":
                        cfg.embedding.model = "BAAI/bge-m3"
                return cfg
            else:
                error("Config file should define a single Config object")
        else:
            error(f"Cannot read config file {config}")

    async def postgres_stores(
        self, uri: str
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

        details: PostgresDetails[MultiEmbedStore] = PostgresDetails(uri)

        vector_store: PGVectorStore = PGVectorStore.from_params(
            connection_string=uri,
            database=details.database,
            host=details.host,
            password=details.password,
            port=str(details.port),
            user=details.user,
            table_name="vectorstore",
            embed_dim=self.config.embed_dim,
            hybrid_search=self.config.hybrid_search,
            hnsw_kwargs={
                "hnsw_m": self.config.hnsw_m,
                "hnsw_ef_construction": self.config.hnsw_ef_construction,
                "hnsw_ef_search": self.config.hnsw_ef_search,
                "hnsw_dist_method": self.config.hnsw_dist_method,
            },
        )
        return docstore, index_store, vector_store

    async def load_embedding(
        self,
        provider: str,
        model: str,
        timeout: int,
        api_key: str | None,
        api_version: str | None,
        base_url: str | None,
        query_instruction: str | None,
        num_workers: int | None,
        verbose: bool = False,
    ) -> BaseEmbedding | BGEM3Embedding:
        logger.info(f"Load embedding {provider}:{model}")
        if provider == "HuggingFace":
            return HuggingFaceEmbedding(
                model_name=model,
                query_instruction=query_instruction,
                show_progress_bar=verbose,
            )
        elif provider == "Ollama":
            return OllamaEmbedding(
                model_name=model,
                base_url=base_url or "http://localhost:11434",
            )
        elif provider == "LlamaCpp":
            return LlamaCppEmbedding(model_path=model)
        elif provider == "OpenAI":
            return OpenAIEmbedding(
                model_name=model,
                api_key=api_key,
                api_version=api_version,
                api_base=base_url,
                timeout=timeout,
            )
        elif provider == "OpenAILike":
            return OpenAILikeEmbedding(
                model_name=model,
                api_key=api_key or "fake_key",
                api_version=api_version,
                api_base=base_url,
                timeout=timeout,
                num_workers=num_workers,
            )
        elif provider == "BGEM3":
            return BGEM3Embedding()
        else:
            error(f"Embedding model not recognized: {model}")

    async def load_llm(
        self,
        provider: str,
        model: str,
        timeout: int,
        api_key: str | None,
        api_version: str | None,
        base_url: str | None,
        verbose: bool = False,
    ) -> LLM:
        logger.info(f"Load LLM {provider}:{model}")
        if provider == "Ollama":
            return Ollama(
                model=model,
                base_url=base_url or "http://localhost:11434",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                context_window=self.config.context_window,
            )
        elif provider == "OpenAILike":
            return OpenAILike(
                model=model,
                api_base=base_url or "http://localhost:1234/v1",
                api_key=api_key or "fake_key",
                api_version=api_version or "",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                context_window=self.config.context_window,
                reasoning_effort=self.config.reasoning_effort,
                timeout=timeout,
            )
        elif provider == "OpenAI":
            return OpenAI(
                model=model,
                api_key=api_key,
                api_base=base_url,
                api_version=api_version,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                context_window=self.config.context_window,
                reasoning_effort=self.config.reasoning_effort,
                timeout=timeout,
            )
        elif provider == "LlamaCpp":
            return LlamaCPP(
                # model_url=model_url,
                model_path=model,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_tokens,
                context_window=self.config.context_window,
                generate_kwargs={},
                model_kwargs={"n_gpu_layers": self.config.gpu_layers},
                verbose=verbose,
            )
        elif provider == "Perplexity":
            return Perplexity(
                model_name=model,
                api_key=api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                context_window=self.config.context_window,
                reasoning_effort=self.config.reasoning_effort,
                # This will determine if the search component is necessary
                # in this particular context
                enable_search_classifier=True,
                timeout=timeout,
            )
        elif provider == "OpenRouter":
            return OpenRouter(
                model_name=model,
                api_key=api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                context_window=self.config.context_window,
                reasoning_effort=self.config.reasoning_effort,
                timeout=timeout,
            )
        elif provider == "LMStudio":
            return LMStudio(
                model_name=model,
                base_url=base_url or "http://localhost:1234/v1",
                temperature=self.config.temperature,
                context_window=self.config.context_window,
                timeout=timeout,
            )
        else:
            error(f"LLM model not recognized: {model}")

    async def determine_fingerprint(
        self,
        input_files: list[Path],
        embed_model: str,
        embed_dim: int,
    ) -> str:
        logger.info("Determine input files fingerprint")
        fingerprint = [
            collection_hash(input_files),
            hashlib.sha512(embed_model.encode("utf-8")).hexdigest(),
            hashlib.sha512(str(embed_dim).encode("utf-8")).hexdigest(),
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

    async def load_splitter(
        self,
        models: Models,
        splitter_model: str,
    ) -> TransformComponent | NoReturn:
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
        logger.info(f"Load splitter {splitter_model}")
        if splitter_model == "Sentence":
            return SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                include_metadata=True,
            )
        elif splitter_model == "SentenceWindow":
            return SentenceWindowNodeParser.from_defaults(
                window_size=self.config.window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
        elif splitter_model == "Semantic":
            embed_llm = None
            if models.semantic_splitter_embed_llm is not None:
                embed_llm = models.semantic_splitter_embed_llm
            elif models.embed_llm is not None:
                embed_llm = models.embed_llm

            if embed_llm is None:
                error("Semantic splitter needs an embedding model")

            if isinstance(embed_llm, BGEM3Embedding):
                error("Semantic splitter not yet working with BGE-M3")

            return SemanticSplitterNodeParser(
                buffer_size=self.config.buffer_size,
                breakpoint_percentile_threshold=self.config.breakpoint_percentile_threshold,
                embed_model=embed_llm,
                include_metadata=True,
            )
        elif splitter_model == "Code":
            return CodeSplitter(
                language=self.config.code_language,
                chunk_lines=self.config.code_chunk_lines,
                chunk_lines_overlap=self.config.code_chunk_lines_overlap,
                max_chars=self.config.code_max_chars,
            )
        else:
            error(f"Splitting model not recognized: {splitter_model}")

    async def split_documents(
        self,
        models: Models,
        splitter_model: str,
        documents: Iterable[Document],
        questions_answered: int | None,
        num_workers: int | None,
    ) -> Sequence[BaseNode]:
        logger.info("Split documents")

        transformations = [
            await self.load_splitter(models, splitter_model=splitter_model)
        ]

        if models.llm is not None:
            transformations.extend(
                [
                    KeywordExtractor(
                        keywords=5,
                        llm=models.keywords_llm or models.llm,
                        num_workers=self.config.num_workers,
                    ),
                    SummaryExtractor(
                        summaries=["self"],
                        llm=models.metadata_extractor_llm or models.llm,
                        num_workers=self.config.num_workers,
                    ),
                    TitleExtractor(
                        nodes=5,
                        llm=models.metadata_extractor_llm or models.llm,
                        num_workers=self.config.num_workers,
                    ),
                ]
            )
            if questions_answered is not None:
                logger.info(f"Generate {questions_answered} questions/chunk")
                transformations.append(
                    QuestionsAnsweredExtractor(
                        questions=questions_answered,
                        llm=models.questions_answered_llm or models.llm,
                        num_workers=self.config.num_workers,
                    )
                )

        pipeline = IngestionPipeline(transformations=transformations)
        return await pipeline.arun(documents=documents, num_workers=num_workers)

    async def populate_vector_store(
        self,
        models: Models,
        nodes: Sequence[BaseNode],
        collect_keywords: bool,
        storage_context: StorageContext,
        verbose: bool = False,
    ) -> tuple[VectorStoreIndex | BGEM3Index, BaseKeywordTableIndex | None]:
        logger.info("Populate vector store")

        index_structs = storage_context.index_store.index_structs()
        for struct in index_structs:
            storage_context.index_store.delete_index_struct(key=struct.index_id)

        if isinstance(models.embed_llm, BGEM3Embedding):
            vector_index = BGEM3Index(
                nodes,
                storage_context=storage_context,
                weights_for_different_modes=[0.4, 0.2, 0.4],
                show_progress=verbose,
            )
        else:
            vector_index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=models.embed_llm,
                show_progress=verbose,
                use_async=True,
            )

        if collect_keywords:
            if models.llm is None:
                keyword_index = SimpleKeywordTableIndex(
                    nodes, storage_context=storage_context
                )
            else:
                keyword_index = KeywordTableIndex(
                    nodes,
                    storage_context=storage_context,
                    llm=models.keywords_llm or models.llm,
                )
        else:
            keyword_index = None

        return vector_index, keyword_index

    # global Settings
    # Settings.llm = models.llm
    # Settings.embed_model = models.embed_llm
    # Settings.chunk_size = self.config.chunk_size
    # Settings.chunk_overlap = self.config.chunk_overlap

    async def index_files(
        self,
        models: Models,
        input_files: list[Path] | None,
        splitter_model: str,
        collect_keywords: bool,
        questions_answered: int | None,
        num_workers: int | None,
        verbose: bool = False,
    ) -> tuple[
        VectorStoreIndex | BGEM3Index | None,
        BaseKeywordTableIndex | None,
    ]:
        persist_dir: Path | None = None
        vector_index: VectorStoreIndex | BGEM3Index | None = None
        keyword_index: BaseKeywordTableIndex | None = None

        if input_files is None:
            logger.info("No input files")
        elif self.config.embedding is None:
            logger.info("No embedding model")
        else:
            logger.info(f"{len(input_files)} input file(s)")
            fp: str = await self.determine_fingerprint(
                input_files, self.config.embedding.model, self.config.embed_dim
            )
            logger.info(f"Fingerprint = {fp}")
            persist_dir = cache_dir(fp)
            self.fingerprint = fp

        if self.config.db_conn is not None:
            docstore, index_store, vector_store = await self.postgres_stores(
                uri=self.config.db_conn
            )
        elif persist_dir is not None and os.path.isdir(persist_dir):
            docstore = SimpleDocumentStore.from_persist_dir(str(persist_dir))
            index_store = SimpleIndexStore.from_persist_dir(str(persist_dir))
            vector_store = SimpleVectorStore.from_persist_dir(str(persist_dir))
        else:
            docstore = SimpleDocumentStore()
            index_store = SimpleIndexStore()
            vector_store = SimpleVectorStore()

        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store,
            vector_store=vector_store,
            persist_dir=(
                str(persist_dir) if persist_dir is not None else DEFAULT_PERSIST_DIR
            ),
        )

        persisted = persist_dir is not None and os.path.isdir(persist_dir)

        if self.config.db_conn is not None or persisted:
            try:
                if self.config.db_conn is not None:
                    logger.info("Read indices from database")
                else:
                    logger.info("Read indices from cache")

                if isinstance(models.embed_llm, BGEM3Embedding):
                    if self.config.db_conn is not None:
                        error("BGE-M3 not current compatible with databases")
                        # logger.info("Read BGE-M3 index from database")
                        # self.vector_index = BGEM3Embedding.load_from_postgres(
                        #     uri=self.config.db_conn,
                        #     storage_context=self.storage_context,
                        #     weights_for_different_modes=[0.4, 0.2, 0.4],
                        # )
                    else:
                        logger.info("Read BGE-M3 index from cache")
                        vector_index = BGEM3Index.load_from_disk(
                            persist_dir=str(persist_dir),
                            weights_for_different_modes=[0.4, 0.2, 0.4],
                        )
                else:
                    indices: IndexList = load_indices_from_storage(
                        storage_context=storage_context,
                        embed_model=(
                            models.embed_llm
                            if not isinstance(models.embed_llm, BGEM3Embedding)
                            else None
                        ),
                        llm=models.llm,
                        index_ids=(
                            ["vector_index", "keyword_index"]
                            if collect_keywords
                            else ["vector_index"]
                        ),
                    )
                    if collect_keywords:
                        [vi, ki] = indices
                        vector_index = cast(VectorStoreIndex, vi)
                        keyword_index = cast(BaseKeywordTableIndex, ki)
                    else:
                        [vi] = indices
                        vector_index = cast(VectorStoreIndex, vi)
                        keyword_index = None

            except ValueError:
                logger.info("Failed to read indices")

        if input_files is not None and not persisted:
            documents = await self.read_documents(
                input_files, num_workers=self.config.num_workers
            )

            nodes = await self.split_documents(
                models,
                splitter_model,
                documents,
                questions_answered=questions_answered,
                num_workers=num_workers,
            )

            vector_index, keyword_index = await self.populate_vector_store(
                models,
                nodes,
                collect_keywords=collect_keywords,
                storage_context=storage_context,
                verbose=verbose,
            )

            vector_index.set_index_id("vector_index")
            if keyword_index is not None:
                keyword_index.set_index_id("keyword_index")

            await self.save_indices(
                storage_context, vector_index, persist_dir=persist_dir
            )

        return (vector_index, keyword_index)

    async def save_indices(
        self,
        storage_context: StorageContext,
        vector_index: VectorStoreIndex | BGEM3Index,
        persist_dir: Path | None,
    ):
        if self.config.db_conn is not None:
            if isinstance(vector_index, BGEM3Index):
                logger.info("Persist BGE-M3 index to database")
                BGEM3Embedding.persist_to_postgres(
                    self.config.db_conn,
                    vector_index,
                )
        elif persist_dir is not None:
            if isinstance(vector_index, BGEM3Index):
                logger.info("Persist storage context and BGE-M3 index")
                vector_index.persist(persist_dir=str(persist_dir))
            logger.info("Persist storage context to disk")
            storage_context.persist(  # pyright: ignore[reportUnknownMemberType]
                persist_dir=str(persist_dir)
            )

    async def load_retriever(
        self,
        models: Models,
        indices: tuple[
            VectorStoreIndex | BGEM3Index | None,
            BaseKeywordTableIndex | None,
        ],
    ) -> BaseRetriever | None:
        vector_index, keyword_index = indices
        vector_retriever, keyword_retriever = None, None

        if vector_index is not None:
            if self.config.db_conn is not None and self.config.hybrid_search:
                logger.info("Create fusion vector retriever")
                vector_retriever = vector_index.as_retriever(
                    vector_store_query_mode="default",
                    similarity_top_k=5,
                )
                text_retriever = vector_index.as_retriever(
                    vector_store_query_mode="sparse",
                    similarity_top_k=5,  # interchangeable with sparse_top_k
                )
                # jww (2025-05-08): Make more of these configurable
                vector_retriever = QueryFusionRetriever(
                    [vector_retriever, text_retriever],
                    similarity_top_k=self.config.top_k,
                    num_queries=1,  # set this to 1 to disable query generation
                    mode=FUSION_MODES.RELATIVE_SCORE,
                    llm=models.llm,
                    use_async=True,
                )
            else:
                logger.info("Create vector retriever")
                vector_retriever = vector_index.as_retriever(
                    similarity_top_k=self.config.top_k
                )
        if keyword_index is not None:
            logger.info("Create keyword retriever")
            keyword_retriever = keyword_index.as_retriever()

        if vector_retriever is not None:
            if keyword_retriever is not None:
                logger.info("Create aggregate custom retriever")
                retriever = CustomRetriever(
                    vector_retriever=vector_retriever,
                    keyword_retriever=keyword_retriever,
                )
            else:
                retriever = vector_retriever
        else:
            if keyword_retriever is not None:
                retriever = keyword_retriever
            else:
                retriever = None

        return retriever

    async def retrieve_nodes(
        self, retriever: BaseRetriever, text: str
    ) -> list[dict[str, Any]]:
        logger.info("Retrieve nodes from vector index")
        nodes = await retriever.aretrieve(text)
        logger.info(f"{len(nodes)} nodes found in vector index")
        return [
            {
                "text": node.text,
                "metadata": node.metadata,
            }
            for node in nodes
        ]

    # Query a document collection
    async def query(
        self,
        models: Models,
        retriever: BaseRetriever | None,
        query: str,
        streaming: bool = False,
        retries: bool = False,
        source_retries: bool = False,
    ) -> RESPONSE_TYPE | NoReturn:
        if retriever is None:
            error("There is no retriever configured to query")
        if models.llm is None:
            error("There is no LLM configured to chat with")

        logger.info("Query with retriever query engine")
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=models.llm,
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
            relevancy_evaluator = RelevancyEvaluator(
                llm=models.evaluator_llm or models.llm,
            )
            # jww (2025-05-04): Allow using different evaluators
            _guideline_evaluator = GuidelineEvaluator(
                llm=models.evaluator_llm or models.llm,
                guidelines=DEFAULT_GUIDELINES
                + "\nThe response should not be overly long.\n"
                + "The response should try to summarize where possible.\n",
            )
            if source_retries:
                logger.info("Add retry source query engine")
                query_engine = RetrySourceQueryEngine(
                    query_engine,
                    evaluator=relevancy_evaluator,
                    llm=models.llm,
                )
            else:
                logger.info("Add retry query engine")
                query_engine = RetryQueryEngine(
                    query_engine, evaluator=relevancy_evaluator
                )

        logger.info("Submit query to LLM")
        return await query_engine.aquery(query)

    async def reset_chat(self):
        self.chat_engine = None
        self.chat_memory = None

    # Chat with the LLM, possibly in the context of a document collection
    async def chat(
        self,
        models: Models,
        retriever: BaseRetriever | None,
        user: str,
        query: str,
        token_limit: int,
        chat_store: SimpleChatStore | None = None,
        summarize_chat: bool = False,
        streaming: bool = False,
    ) -> StreamingAgentChatResponse | AgentChatResponse | NoReturn:
        if models.llm is None:
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
                    llm=models.llm,
                )
            else:
                self.chat_memory = ChatMemoryBuffer.from_defaults(  # pyright: ignore[reportUnknownMemberType]
                    token_limit=token_limit,
                    chat_store=chat_store,
                    chat_store_key=user,
                    llm=models.llm,
                )

        if self.chat_engine is None:
            if retriever is None:
                self.chat_engine = SimpleChatEngine.from_defaults(
                    llm=models.llm,
                    memory=self.chat_memory,
                    system_prompt="You are a helpful AI assistant.",
                )
            else:
                self.chat_engine = ContextChatEngine.from_defaults(
                    retriever=retriever,
                    llm=models.llm,
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


def query_perplexity(query: str) -> str:
    """
    Queries the Perplexity API via the LlamaIndex integration.

    This function instantiates a Perplexity LLM with updated default settings
    (using model "sonar-pro" and enabling search classifier so that the API
    can intelligently decide if a search is needed), wraps the query into a
    ChatMessage, and returns the generated response content.
    """
    pplx_api_key = "your-perplexity-api-key"  # Replace with your actual API key

    llm = Perplexity(
        api_key=pplx_api_key,
        model="sonar-pro",
        temperature=0.7,
    )

    messages = [ChatMessage(role="user", content=query)]
    response = llm.chat(messages)
    return response.message.content or ""


def query_perplexity_tool() -> FunctionTool:
    return FunctionTool.from_defaults(fn=query_perplexity)


async def rag_initialize(
    args: Args,
) -> tuple[RAGWorkflow, Models, BaseRetriever | None]:
    config = await RAGWorkflow.load_config(Path(args.config))

    if args.from_:
        input_files = read_files(args.from_, config.recursive)
    else:
        input_files = awaitable_none()

    rag = RAGWorkflow(config)

    models = await rag.initialize(
        verbose=args.verbose,
    )
    indices = await rag.index_files(
        models=models,
        input_files=await input_files,
        splitter_model=config.splitter,
        collect_keywords=config.collect_keywords,
        questions_answered=config.questions_answered,
        num_workers=config.num_workers,
        verbose=args.verbose,
    )
    retriever = await rag.load_retriever(models, indices)

    return (rag, models, retriever)


def rebuild_postgres_db(db_name: str):
    connection_string = "postgresql://postgres:password@localhost:5432"
    with psycopg2.connect(connection_string) as conn:
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {db_name}")
            c.execute(f"CREATE DATABASE {db_name}")

    connection_string2 = "postgresql://postgres:password@localhost:5432/{db_name}"
    with psycopg2.connect(connection_string2) as conn:
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute("CREATE EXTENSION vector;")


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
