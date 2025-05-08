# pyright: reportMissingTypeStubs=false
# pyright: reportExplicitAny=false

import asyncio
import base64
import hashlib
import logging
import os
import sys
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
import psycopg2
import numpy as np

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.response_synthesizers import ResponseMode

from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from orgparse.node import OrgNode
from pathlib import Path
from typed_argparse import TypedArgs
from typing import (
    Any,
    Literal,
    NoReturn,
    cast,
    final,
    no_type_check,
    override,
    TypeVar,
    Generic,
)
from xdg_base_dirs import xdg_cache_home
from urllib.parse import urlparse

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
from llama_index.core.tools import FunctionTool
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
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import (
    BaseNode,
    Document,
    Node,
    NodeWithScore,
    TransformComponent,
)
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.vector_stores.simple import SimpleVectorStore

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.perplexity import Perplexity
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.openrouter import OpenRouter
from llama_index.indices.managed.bge_m3 import BGEM3Index
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.index_store.postgres import PostgresIndexStore

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
            if await asyncio.to_thread(os.path.isfile, os.path.join(directory, f))
        ]


async def read_files(read_from: str, recursive: bool = False) -> list[Path] | NoReturn:
    if read_from == "-":
        # Reading from stdin is still blocking; consider using asyncio streams if needed
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


class Args(TypedArgs):
    config: str | None
    verbose: bool
    db_conn: str | None
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    hnsw_dist_method: str
    embed_provider: str | None
    embed_model: str | None
    embed_api_key: str | None
    embed_api_version: str | None
    embed_base_url: str | None
    embed_dim: int
    query_instruction: str | None
    chunk_size: int
    chunk_overlap: int
    splitter: str
    semantic_splitter_embed_provider: str | None
    semantic_splitter_embed_model: str | None
    semantic_splitter_embed_api_key: str | None
    semantic_splitter_embed_api_version: str | None
    semantic_splitter_embed_base_url: str | None
    semantic_splitter_query_instruction: str | None
    buffer_size: int
    breakpoint_percentile_threshold: int
    window_size: int
    questions_answered: int | None
    questions_answered_provider: str | None
    questions_answered_model: str | None
    questions_answered_api_key: str | None
    questions_answered_api_version: str | None
    questions_answered_base_url: str | None
    hybrid_search: bool
    top_k: int
    llm_provider: str | None
    llm_model: str | None
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
    metadata_extractor_provider: str | None
    metadata_extractor_model: str | None
    metadata_extractor_api_key: str | None
    metadata_extractor_api_version: str | None
    metadata_extractor_base_url: str | None
    collect_keywords: bool
    keywords_provider: str | None
    keywords_model: str | None
    keywords_api_key: str | None
    keywords_api_version: str | None
    keywords_base_url: str | None
    retries: bool
    source_retries: bool
    evaluator_provider: str | None
    evaluator_model: str | None
    evaluator_api_key: str | None
    evaluator_api_version: str | None
    evaluator_base_url: str | None
    summarize_chat: bool
    num_workers: int
    host: str
    port: int
    reload_server: bool
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
                    error(f"Data not available in table {tablename} row {row_id}")

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
class RAGWorkflow:
    verbose: bool = False
    fingerprint: str | None = None

    embed_llm: BaseEmbedding | BGEM3Embedding | None = None
    semantic_splitter_embed_llm: BaseEmbedding | BGEM3Embedding | None = None
    llm: LLM | None = None
    questions_answered_llm: LLM | None = None
    metadata_extractor_llm: LLM | None = None
    keywords_llm: LLM | None = None
    evaluator_llm: LLM | None = None

    storage_context: StorageContext | None = None
    vector_retriever: BaseRetriever | None = None
    keyword_retriever: BaseRetriever | None = None
    retriever: BaseRetriever | None = None

    vector_index: VectorStoreIndex | BGEM3Index | None = None
    keyword_index: BaseKeywordTableIndex | None = None

    chat_memory: ChatMemoryBuffer | ChatSummaryMemoryBuffer | None = None
    chat_history: list[ChatMessage] | None = None
    chat_engine: BaseChatEngine | None = None

    async def initialize(self, args: Args):
        embed_llm = (
            self.load_embedding(
                args.embed_provider,
                args.embed_model,
                args.timeout,
                args.embed_api_key,
                args.embed_api_version,
                args.embed_base_url,
                args.query_instruction,
            )
            if args.embed_provider and args.embed_model
            else awaitable_none()
        )
        semantic_splitter_embed_llm = (
            self.load_embedding(
                args.semantic_splitter_embed_provider,
                args.semantic_splitter_embed_model,
                args.timeout,
                args.semantic_splitter_embed_api_key,
                args.semantic_splitter_embed_api_version,
                args.semantic_splitter_embed_base_url,
                args.semantic_splitter_query_instruction,
            )
            if args.semantic_splitter_embed_provider
            and args.semantic_splitter_embed_model
            else awaitable_none()
        )
        llm = (
            self.load_llm(
                args.llm_provider,
                args.llm_model,
                args.timeout,
                args.llm_api_key,
                args.llm_api_version,
                args.llm_base_url,
                args,
            )
            if args.llm_provider and args.llm_model
            else awaitable_none()
        )
        questions_answered_llm = (
            self.load_llm(
                args.questions_answered_provider,
                args.questions_answered_model,
                args.timeout,
                args.questions_answered_api_key,
                args.questions_answered_api_version,
                args.questions_answered_base_url,
                args,
            )
            if args.questions_answered_provider and args.questions_answered_model
            else awaitable_none()
        )
        metadata_extractor_llm = (
            self.load_llm(
                args.metadata_extractor_provider,
                args.metadata_extractor_model,
                args.timeout,
                args.metadata_extractor_api_key,
                args.metadata_extractor_api_version,
                args.metadata_extractor_base_url,
                args,
            )
            if args.metadata_extractor_provider and args.metadata_extractor_model
            else awaitable_none()
        )
        keywords_llm = (
            self.load_llm(
                args.keywords_provider,
                args.keywords_model,
                args.timeout,
                args.keywords_api_key,
                args.keywords_api_version,
                args.keywords_base_url,
                args,
            )
            if args.keywords_provider and args.keywords_model
            else awaitable_none()
        )
        evaluator_llm = (
            self.load_llm(
                args.evaluator_provider,
                args.evaluator_model,
                args.timeout,
                args.evaluator_api_key,
                args.evaluator_api_version,
                args.evaluator_base_url,
                args,
            )
            if args.evaluator_provider and args.evaluator_model
            else awaitable_none()
        )

        self.embed_llm = await embed_llm
        self.semantic_splitter_embed_llm = await semantic_splitter_embed_llm
        self.llm = await llm
        self.questions_answered_llm = await questions_answered_llm
        self.metadata_extractor_llm = await metadata_extractor_llm
        self.keywords_llm = await keywords_llm
        self.evaluator_llm = await evaluator_llm

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

        details: PostgresDetails[MultiEmbedStore] = PostgresDetails(uri)

        vector_store: PGVectorStore = PGVectorStore.from_params(
            connection_string=uri,
            database=details.database,
            host=details.host,
            password=details.password,
            port=str(details.port),
            user=details.user,
            table_name="vectorstore",
            embed_dim=args.embed_dim,
            hybrid_search=args.hybrid_search,
            hnsw_kwargs={
                "hnsw_m": args.hnsw_m,
                "hnsw_ef_construction": args.hnsw_ef_construction,
                "hnsw_ef_search": args.hnsw_ef_search,
                "hnsw_dist_method": args.hnsw_dist_method,
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
    ) -> BaseEmbedding | BGEM3Embedding:
        logger.info(f"Load embedding {provider}:{model}")
        if provider == "HuggingFace":
            return HuggingFaceEmbedding(
                model_name=model,
                query_instruction=query_instruction,
                show_progress_bar=self.verbose,
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
        args: Args,
    ) -> LLM:
        logger.info(f"Load LLM {provider}:{model}")
        if provider == "Ollama":
            return Ollama(
                model=model,
                base_url=base_url or "http://localhost:11434",
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                context_window=args.context_window,
            )
        elif provider == "OpenAILike":
            return OpenAILike(
                model=model,
                api_base=base_url or "http://localhost:1234/v1",
                api_key=api_key or "fake_key",
                api_version=api_version or "",
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                context_window=args.context_window,
                reasoning_effort=args.reasoning_effort,
                timeout=timeout,
            )
        elif provider == "OpenAI":
            return OpenAI(
                model=model,
                api_key=api_key,
                api_base=base_url,
                api_version=api_version,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                context_window=args.context_window,
                reasoning_effort=args.reasoning_effort,
                timeout=timeout,
            )
        elif provider == "LlamaCpp":
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
        elif provider == "Perplexity":
            return Perplexity(
                model_name=model,
                api_key=api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                context_window=args.context_window,
                reasoning_effort=args.reasoning_effort,
                # This will determine if the search component is necessary
                # in this particular context
                enable_search_classifier=True,
                timeout=timeout,
            )
        elif provider == "OpenRouter":
            return OpenRouter(
                model_name=model,
                api_key=api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                context_window=args.context_window,
                reasoning_effort=args.reasoning_effort,
                timeout=timeout,
            )
        elif provider == "LMStudio":
            return LMStudio(
                model_name=model,
                base_url=base_url or "http://localhost:1234/v1",
                temperature=args.temperature,
                context_window=args.context_window,
                timeout=timeout,
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

    async def load_splitter(
        self, model: str, args: Args
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
            embed_llm = None
            if self.semantic_splitter_embed_llm is not None:
                embed_llm = self.semantic_splitter_embed_llm
            elif self.embed_llm is not None:
                embed_llm = self.embed_llm

            if embed_llm is None:
                error("Semantic splitter needs an embedding model")

            if isinstance(embed_llm, BGEM3Embedding):
                error("Semantic splitter not yet working with BGE-M3")

            return SemanticSplitterNodeParser(
                buffer_size=args.buffer_size,
                breakpoint_percentile_threshold=args.breakpoint_percentile_threshold,
                embed_model=embed_llm,
                include_metadata=True,
            )
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
        logger.info("Split documents")

        transformations = [await self.load_splitter(split_model, args)]

        if self.llm is not None:
            transformations.extend(
                [
                    KeywordExtractor(keywords=5, llm=self.keywords_llm or self.llm),
                    SummaryExtractor(
                        summaries=["self"], llm=self.metadata_extractor_llm or self.llm
                    ),
                    TitleExtractor(
                        nodes=5, llm=self.metadata_extractor_llm or self.llm
                    ),
                ]
            )
            if questions_answered is not None:
                logger.info(f"Generate {questions_answered} questions for each chunk")
                transformations.append(
                    QuestionsAnsweredExtractor(
                        questions=questions_answered,
                        llm=self.questions_answered_llm or self.llm,
                    )
                )

        pipeline = IngestionPipeline(transformations=transformations)
        return await pipeline.arun(documents=documents, num_workers=num_workers)

    async def populate_vector_store(
        self,
        nodes: Sequence[BaseNode],
        collect_keywords: bool,
        storage_context: StorageContext,
    ) -> tuple[VectorStoreIndex | BGEM3Index, BaseKeywordTableIndex | None]:
        logger.info("Populate vector store")

        index_structs = storage_context.index_store.index_structs()
        for struct in index_structs:
            storage_context.index_store.delete_index_struct(key=struct.index_id)

        if isinstance(self.embed_llm, BGEM3Embedding):
            vector_index = BGEM3Index(
                nodes,
                storage_context=storage_context,
                weights_for_different_modes=[0.4, 0.2, 0.4],
                show_progress=self.verbose,
            )
        else:
            vector_index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_llm,
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
                    llm=self.keywords_llm or self.llm,
                )
        else:
            keyword_index = None

        return vector_index, keyword_index

    # global Settings
    # Settings.llm = self.llm
    # Settings.embed_model = self.embed_llm
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
            try:
                if args.db_conn is not None:
                    logger.info("Read indices from database")
                else:
                    logger.info("Read indices from cache")

                if isinstance(self.embed_llm, BGEM3Embedding):
                    if args.db_conn is not None:
                        error("BGE-M3 not current compatible with databases")
                        logger.info("Read BGE-M3 index from database")
                        self.vector_index = BGEM3Embedding.load_from_postgres(
                            uri=args.db_conn,
                            storage_context=self.storage_context,
                            weights_for_different_modes=[0.4, 0.2, 0.4],
                        )
                    else:
                        logger.info("Read BGE-M3 index from cache")
                        self.vector_index = BGEM3Index.load_from_disk(
                            persist_dir=str(persist_dir),
                            weights_for_different_modes=[0.4, 0.2, 0.4],
                        )
                else:
                    indices: IndexList = (  # pyright: ignore[reportUnknownVariableType]
                        load_indices_from_storage(
                            storage_context=self.storage_context,
                            embed_model=(
                                self.embed_llm
                                if not isinstance(self.embed_llm, BGEM3Embedding)
                                else None
                            ),
                            llm=self.llm,
                            index_ids=(
                                ["vector_index", "keyword_index"]
                                if collect_keywords
                                else ["vector_index"]
                            ),
                        )
                    )
                    if collect_keywords:
                        [vector_index, keyword_index] = indices
                        self.vector_index = cast(VectorStoreIndex, vector_index)
                        self.keyword_index = cast(BaseKeywordTableIndex, keyword_index)
                    else:
                        [vector_index] = indices
                        self.vector_index = cast(VectorStoreIndex, vector_index)
                        self.keyword_index = None

            except ValueError:
                logger.info("Failed to read indices")
                self.vector_index = None
                self.keyword_index = None

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

            await self.save_indices(persist_dir, args)

    async def save_indices(
        self,
        persist_dir: Path | None,
        args: Args,  # jww (2025-05-04): Should not be here
    ):
        if args.db_conn is not None:
            if self.vector_index is not None:
                if isinstance(self.vector_index, BGEM3Index):
                    logger.info("Persist BGE-M3 index to database")
                    BGEM3Embedding.persist_to_postgres(args.db_conn, self.vector_index)
        elif persist_dir is not None:
            if self.vector_index is not None:
                self.vector_index.set_index_id("vector_index")
                if self.keyword_index is not None:
                    self.keyword_index.set_index_id("keyword_index")

                if isinstance(self.vector_index, BGEM3Index):
                    logger.info("Persist storage context and BGE-M3 index")
                    self.vector_index.persist(persist_dir=str(persist_dir))
                elif self.storage_context is not None:
                    logger.info("Persist storage context to disk")
                    self.storage_context.persist(  # pyright: ignore[reportUnknownMemberType]
                        persist_dir=str(persist_dir)
                    )

    async def load_retriever(self, args: Args):
        if self.vector_index is not None:
            if args.db_conn is not None and args.hybrid_search:
                logger.info("Create fusion vector retriever")
                vector_retriever = self.vector_index.as_retriever(
                    vector_store_query_mode="default",
                    similarity_top_k=5,
                )
                text_retriever = self.vector_index.as_retriever(
                    vector_store_query_mode="sparse",
                    similarity_top_k=5,  # interchangeable with sparse_top_k in this context
                )
                # jww (2025-05-08): Make more of these configurable
                self.vector_retriever = QueryFusionRetriever(
                    [vector_retriever, text_retriever],
                    similarity_top_k=args.top_k,
                    num_queries=1,  # set this to 1 to disable query generation
                    mode=FUSION_MODES.RELATIVE_SCORE,
                    llm=self.llm,
                    use_async=True,
                )
            else:
                logger.info("Create vector retriever")
                self.vector_retriever = self.vector_index.as_retriever(
                    similarity_top_k=args.top_k
                )
        if self.keyword_index is not None:
            logger.info("Create keyword retriever")
            self.keyword_retriever = self.keyword_index.as_retriever()

        if self.vector_retriever is not None:
            if self.keyword_retriever is not None:
                logger.info("Create aggregate custom retriever")
                self.retriever = CustomRetriever(
                    vector_retriever=self.vector_retriever,
                    keyword_retriever=self.keyword_retriever,
                )
            else:
                self.retriever = self.vector_retriever
        else:
            if self.keyword_retriever is not None:
                self.retriever = self.keyword_retriever

    async def retrieve_nodes(self, text: str) -> list[dict[str, Any]]:
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
            relevancy_evaluator = RelevancyEvaluator(llm=self.evaluator_llm or self.llm)
            # jww (2025-05-04): Allow using different evaluators
            _guideline_evaluator = GuidelineEvaluator(
                llm=self.evaluator_llm or self.llm,
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


def query_perplexity(query: str) -> str:
    """
    Queries the Perplexity API via the LlamaIndex integration.

    This function instantiates a Perplexity LLM with updated default settings
    (using model "sonar-pro" and enabling search classifier so that the API can
    intelligently decide if a search is needed), wraps the query into a ChatMessage,
    and returns the generated response content.
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


async def rag_initialize(args: Args) -> RAGWorkflow:
    if args.from_:
        input_files = read_files(args.from_, args.recursive)
    else:
        input_files = awaitable_none()

    rag = RAGWorkflow(verbose=args.verbose)

    await rag.initialize(args)
    await rag.index_files(
        input_files=await input_files,
        splitter=args.splitter,
        collect_keywords=args.collect_keywords,
        questions_answered=args.questions_answered,
        num_workers=args.num_workers,
        args=args,
    )
    await rag.load_retriever(args=args)

    return rag


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
