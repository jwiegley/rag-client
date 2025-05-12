# pyright: reportMissingTypeStubs=false
# pyright: reportExplicitAny=false
# pyright: reportAny=false

import asyncio
import base64
import hashlib
import logging
import os
import sys
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
import psycopg2
import llama_cpp

from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from dataclass_wizard import JSONWizard, YAMLWizard
from functools import cache
from pathlib import Path
from urllib.parse import urlparse
from orgparse.node import OrgNode
from xdg_base_dirs import xdg_cache_home
from typing import (
    Any,
    Literal,
    NoReturn,
    cast,
    final,
    no_type_check,
    override,
)

from llama_index.core import (
    KeywordTableIndex,
    QueryBundle,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
    VectorStoreIndex,
    load_indices_from_storage,  # pyright: ignore[reportUnknownVariableType]
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
    JSONNodeParser,
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
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.index_store.postgres import PostgresIndexStore
from llama_index.vector_stores.postgres import PGVectorStore


# Utility functions


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
        input_files = [
            Path(line.strip()) for line in sys.stdin if line.strip()
        ]
        if not input_files:
            error("No filenames provided on standard input")
        return input_files
    elif await asyncio.to_thread(os.path.isdir, read_from):
        return await list_files(Path(read_from), recursive)
    elif await asyncio.to_thread(os.path.isfile, read_from):
        return [Path(read_from)]
    else:
        error(f"Input path is unrecognized or non-existent: {read_from}")


async def convert_str(read_from: str | None) -> str | None:
    if read_from is None:
        return read_from
    elif read_from == "-":
        s = sys.stdin.read()
        if not s:
            error("No input provided on standard input")
        return s
    elif await asyncio.to_thread(os.path.isfile, read_from):
        with open(read_from, "r") as f:
            return f.read()
    else:
        return read_from


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


# Config


class GlobalJSONMeta(JSONWizard.Meta):
    tag_key = "type"  # pyright: ignore[reportUnannotatedClassAttribute]
    auto_assign_tags = True  # pyright: ignore[reportUnannotatedClassAttribute]


@dataclass
class LLMConfig(YAMLWizard):
    provider: str
    model: str
    base_url: str | None = None
    api_key: str = "fake_key"
    api_version: str = ""
    timeout: int = 60
    system_prompt: str | None = None
    temperature: float = 1.0
    max_tokens: int = 200
    context_window: int = 2048
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    gpu_layers: int = -1


@dataclass
class EmbeddingConfig(LLMConfig):
    dimensions: int = 512
    query_instruction: str | None = None


@dataclass
class SentenceSplitterConfig(YAMLWizard):
    chunk_size: int = 512
    chunk_overlap: int = 20
    include_metadata: bool = True


@dataclass
class SentenceWindowSplitterConfig(YAMLWizard):
    window_size: int = 3
    window_metadata_key: str = "window"
    original_text_metadata_key: str = "original_text"


@dataclass
class SemanticSplitterConfig(YAMLWizard):
    embedding: EmbeddingConfig | None
    buffer_size: int = 256
    breakpoint_percentile_threshold: int = 95
    include_metadata: bool = True


@dataclass
class JSONNodeParserConfig(YAMLWizard):
    include_metadata: bool = True
    include_prev_next_rel: bool = True


@dataclass
class CodeSplitterConfig(YAMLWizard):
    language: str = "python"
    chunk_lines: int = 40
    chunk_lines_overlap: int = 15
    max_chars: int = 1500


@dataclass
class Config(YAMLWizard):
    db_conn: str | None = None
    embedding: EmbeddingConfig | None = None
    llm: LLMConfig | None = None
    splitter: (
        SentenceSplitterConfig
        | SentenceWindowSplitterConfig
        | SemanticSplitterConfig
        | JSONNodeParserConfig
        | CodeSplitterConfig
    ) = field(default_factory=SentenceSplitterConfig)
    # Questions answered extractor
    questions_answered: int | None = None
    questions_answered_llm: LLMConfig | None = None
    hybrid_search: bool = False
    chat_user: str | None = None
    metadata_extractor_llm: LLMConfig | None = None
    collect_keywords: bool = False
    keywords_llm: LLMConfig | None = None
    retries: bool = False
    source_retries: bool = False
    evaluator_llm: LLMConfig | None = None
    summarize_chat: bool = False
    num_workers: int = DEFAULT_NUM_WORKERS
    streaming: bool = False
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    hnsw_dist_method: str = "vector_cosine_ops"
    top_k: int = 3
    token_limit: int = 1500
    recursive: bool = False
    host: str = "localhost"
    port: int = 8000
    reload_server: bool = False


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

        # In orgparse, list(org_content) ALL the nodes in the file So we use
        # this to process the nodes below the split_depth as separate
        # documents and skip the rest. This means at a split_depth of 2, we
        # make documents from nodes at levels 0 (whole file), 1, and 2. The
        # text will be present in multiple documents!
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

        self._model = llama_cpp.Llama(
            model_path=model_path,
            embedding=True,
            n_gpu_layers=-1,
            split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
            main_gpu=0,
            tensor_split=None,
            rpc_servers=None,
            vocab_only=False,
            use_mmap=True,
            use_mlock=False,
            kv_overrides=None,
            seed=llama_cpp.LLAMA_DEFAULT_SEED,
            n_ctx=512,
            n_batch=512,
            n_ubatch=512,
            n_threads=None,
            n_threads_batch=None,
            rope_scaling_type=llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,
            rope_freq_base=0.0,
            rope_freq_scale=0.0,
            yarn_ext_factor=-1.0,
            yarn_attn_factor=1.0,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            yarn_orig_ctx=0,
            logits_all=False,
            offload_kqv=True,
            flash_attn=False,
            no_perf=False,
            last_n_tokens_size=64,
            lora_base=None,
            lora_scale=1.0,
            lora_path=None,
            numa=False,
            chat_format=None,
            chat_handler=None,
            draft_model=None,
            tokenizer=None,
            type_k=None,
            type_v=None,
            spm_infill=False,
            verbose=kwargs["verbose"],
        )

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
    """Custom retriever performing both semantic search and hybrid search."""

    @no_type_check
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        verbose: bool = False,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__(verbose=verbose)

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


class PostgresDetails:
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

    def unpickle_from_table[T](self, tablename: str, row_id: int) -> Any:
        import pickle
        import psycopg2

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
                    return None

                binary_data = row[0]
                if isinstance(binary_data, memoryview):
                    binary_data = binary_data.tobytes()
                return pickle.loads(binary_data)

    def pickle_to_table[U](self, tablename: str, row_id: int, data: object):
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
        details: PostgresDetails = PostgresDetails(uri)
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
        details: PostgresDetails = PostgresDetails(uri)
        details.pickle_to_table(
            "multi_embed_store",
            1,
            index._multi_embed_store,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportPrivateUsage]
        )


@dataclass
class Models:
    embed_llm: BaseEmbedding | BGEM3Embedding | None = None
    llm: LLM | None = None
    questions_answered_llm: LLM | None = None
    metadata_extractor_llm: LLM | None = None
    keywords_llm: LLM | None = None
    evaluator_llm: LLM | None = None


@dataclass
class RAGWorkflow:
    config: Config
    logger: logging.Logger
    chat_memory: ChatMemoryBuffer | ChatSummaryMemoryBuffer | None = None
    chat_history: list[ChatMessage] | None = None
    chat_engine: BaseChatEngine | None = None

    async def initialize(self, verbose: bool = False) -> Models:
        embed_llm = (
            self.__load_embedding(
                self.config.embedding,
                verbose=verbose,
            )
            if self.config.embedding
            else awaitable_none()
        )
        llm = (
            self.__load_llm(
                self.config.llm,
                verbose=verbose,
            )
            if self.config.llm
            else awaitable_none()
        )
        questions_answered_llm = (
            self.__load_llm(
                self.config.questions_answered_llm,
                verbose=verbose,
            )
            if self.config.questions_answered_llm
            else awaitable_none()
        )
        metadata_extractor_llm = (
            self.__load_llm(
                self.config.metadata_extractor_llm,
                verbose=verbose,
            )
            if self.config.metadata_extractor_llm
            else awaitable_none()
        )
        keywords_llm = (
            self.__load_llm(
                self.config.keywords_llm,
                verbose=verbose,
            )
            if self.config.keywords_llm
            else awaitable_none()
        )
        evaluator_llm = (
            self.__load_llm(
                self.config.evaluator_llm,
                verbose=verbose,
            )
            if self.config.evaluator_llm
            else awaitable_none()
        )

        return Models(
            embed_llm=await embed_llm,
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

    async def __postgres_stores(
        self, uri: str, embedding_dimensions: int
    ) -> tuple[PostgresDocumentStore, PostgresIndexStore, PGVectorStore]:
        self.logger.info("Create Postgres store objects")
        docstore: PostgresDocumentStore = PostgresDocumentStore.from_uri(
            uri=uri,
            table_name="docstore",
        )
        index_store: PostgresIndexStore = PostgresIndexStore.from_uri(
            uri=uri,
            table_name="indexstore",
        )

        details: PostgresDetails = PostgresDetails(uri)

        vector_store: PGVectorStore = PGVectorStore.from_params(
            connection_string=uri,
            database=details.database,
            host=details.host,
            password=details.password,
            port=str(details.port),
            user=details.user,
            table_name="vectorstore",
            embed_dim=embedding_dimensions,
            hybrid_search=self.config.hybrid_search,
            hnsw_kwargs={
                "hnsw_m": self.config.hnsw_m,
                "hnsw_ef_construction": self.config.hnsw_ef_construction,
                "hnsw_ef_search": self.config.hnsw_ef_search,
                "hnsw_dist_method": self.config.hnsw_dist_method,
            },
        )
        return docstore, index_store, vector_store

    async def __load_embedding(
        self,
        embed_config: EmbeddingConfig,
        verbose: bool = False,
    ) -> BaseEmbedding | BGEM3Embedding:
        self.logger.info(
            f"Load embedding {embed_config.provider}:{embed_config.model}"
        )
        if embed_config.provider == "HuggingFace":
            return HuggingFaceEmbedding(
                model_name=embed_config.model,
                query_instruction=embed_config.query_instruction,
                show_progress_bar=verbose,
                # text_instruction=None,
                # normalize=True,
                embed_batch_size=DEFAULT_EMBED_BATCH_SIZE,
                # cache_folder=None,
                # trust_remote_code=False,
                parallel_process=True,
            )
        elif embed_config.provider == "Ollama":
            return OllamaEmbedding(
                model_name=embed_config.model,
                base_url=embed_config.base_url or "http://localhost:11434",
                embed_batch_size=DEFAULT_EMBED_BATCH_SIZE,
            )
        elif embed_config.provider == "LlamaCpp":
            return LlamaCppEmbedding(model_path=embed_config.model)
        elif embed_config.provider == "OpenAI":
            return OpenAIEmbedding(
                model_name=embed_config.model,
                api_key=embed_config.api_key,
                api_version=embed_config.api_version,
                api_base=embed_config.base_url,
                timeout=embed_config.timeout,
            )
        elif embed_config.provider == "OpenAILike":
            return OpenAILikeEmbedding(
                model_name=embed_config.model,
                api_key=embed_config.api_key or "fake_key",
                api_version=embed_config.api_version,
                api_base=embed_config.base_url,
                timeout=embed_config.timeout,
            )
        elif embed_config.provider == "BGEM3":
            return BGEM3Embedding()
        else:
            error(f"Embedding model not recognized: {embed_config.model}")

    async def __load_llm(
        self,
        llm_config: LLMConfig,
        verbose: bool = False,
    ) -> LLM:
        self.logger.info(f"Load LLM {llm_config.provider}:{llm_config.model}")
        if llm_config.provider == "Ollama":
            return Ollama(
                model=llm_config.model,
                base_url=llm_config.base_url or "http://localhost:11434",
                temperature=llm_config.temperature,
                context_window=llm_config.context_window,
                max_tokens=llm_config.max_tokens,
                system_prompt=await convert_str(llm_config.system_prompt),
                # request_timeout,
                # prompt_key,
                # json_mode,
                is_function_calling_model=False,
                # keep_alive,
            )
        elif llm_config.provider == "OpenAILike":
            return OpenAILike(
                model=llm_config.model,
                api_base=llm_config.base_url or "http://localhost:1234/v1",
                api_key=llm_config.api_key or "fake_key",
                api_version=llm_config.api_version or "",
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                context_window=llm_config.context_window,
                reasoning_effort=llm_config.reasoning_effort,
                timeout=llm_config.timeout,
                is_chat_model=True,
                is_function_calling_model=False,
                system_prompt=await convert_str(llm_config.system_prompt),
                # max_retries,
                # reuse_client,
            )
        elif llm_config.provider == "OpenAI":
            return OpenAI(
                model=llm_config.model,
                api_key=llm_config.api_key,
                api_base=llm_config.base_url,
                api_version=llm_config.api_version,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                context_window=llm_config.context_window,
                reasoning_effort=llm_config.reasoning_effort,
                timeout=llm_config.timeout,
                system_prompt=await convert_str(llm_config.system_prompt),
                # max_retries,
                # reuse_client,
                # strict,
            )
        elif llm_config.provider == "LlamaCpp":
            return LlamaCPP(
                # model_url=model_url,
                model_path=llm_config.model,
                temperature=llm_config.temperature,
                max_new_tokens=llm_config.max_tokens,
                context_window=llm_config.context_window,
                generate_kwargs={},
                model_kwargs={"n_gpu_layers": llm_config.gpu_layers},
                verbose=verbose,
                system_prompt=await convert_str(llm_config.system_prompt),
            )
        elif llm_config.provider == "Perplexity":
            return Perplexity(
                model_name=llm_config.model,
                api_key=llm_config.api_key,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                context_window=llm_config.context_window,
                reasoning_effort=llm_config.reasoning_effort,
                # This will determine if the search component is necessary
                # in this particular context
                enable_search_classifier=True,
                timeout=llm_config.timeout,
                system_prompt=await convert_str(llm_config.system_prompt),
                # max_retries,
            )
        elif llm_config.provider == "OpenRouter":
            return OpenRouter(
                model_name=llm_config.model,
                api_key=llm_config.api_key,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                context_window=llm_config.context_window,
                reasoning_effort=llm_config.reasoning_effort,
                timeout=llm_config.timeout,
                system_prompt=await convert_str(llm_config.system_prompt),
                is_chat_model=True,
            )
        elif llm_config.provider == "LMStudio":
            return LMStudio(
                model_name=llm_config.model,
                base_url=llm_config.base_url or "http://localhost:1234/v1",
                temperature=llm_config.temperature,
                context_window=llm_config.context_window,
                timeout=llm_config.timeout,
                system_prompt=await convert_str(llm_config.system_prompt),
                is_chat_model=True,
                # request_timeout,
                # num_output,
            )
        else:
            error(f"LLM model not recognized: {llm_config.model}")

    async def __determine_fingerprint(
        self,
        input_files: list[Path],
        embed_model: str,
        embed_dim: int,
    ) -> str:
        self.logger.info("Determine input files fingerprint")
        fingerprint = [
            collection_hash(input_files),
            hashlib.sha512(embed_model.encode("utf-8")).hexdigest(),
            hashlib.sha512(str(embed_dim).encode("utf-8")).hexdigest(),
        ]
        final_hash = "\n".join(fingerprint).encode("utf-8")
        final_base64 = base64.b64encode(final_hash).decode("utf-8")
        return final_base64[0:32]

    async def __read_documents(
        self,
        input_files: list[Path],
        num_workers: int | None,
        verbose: bool = False,
    ) -> Iterable[Document]:
        self.logger.info("Read documents from disk")
        file_extractor: dict[str, BaseReader] = {
            ".org": OrgReader(),
        }
        return SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=file_extractor,
            recursive=self.config.recursive,
        ).load_data(num_workers=num_workers, show_progress=verbose)

    async def __load_splitter(
        self,
        models: Models,
        splitter: (
            SentenceSplitterConfig
            | SentenceWindowSplitterConfig
            | SemanticSplitterConfig
            | JSONNodeParserConfig
            | CodeSplitterConfig
        ),
        verbose: bool = False,
    ) -> TransformComponent | NoReturn:
        self.logger.info(f"Load splitter {splitter.__class__.__name__}")
        if isinstance(splitter, SentenceSplitterConfig):
            return SentenceSplitter(
                chunk_size=splitter.chunk_size,
                chunk_overlap=splitter.chunk_overlap,
                include_metadata=splitter.include_metadata,
            )
        elif isinstance(splitter, SentenceWindowSplitterConfig):
            return SentenceWindowNodeParser.from_defaults(
                window_size=splitter.window_size,
                window_metadata_key=splitter.window_metadata_key,
                original_text_metadata_key=splitter.original_text_metadata_key,
            )
        elif isinstance(splitter, SemanticSplitterConfig):
            if splitter.embedding is not None:
                embed_llm = await self.__load_embedding(
                    splitter.embedding,
                    verbose=verbose,
                )
            elif models.embed_llm is not None:
                embed_llm = models.embed_llm
            else:
                error("Semantic splitter needs embedding model")

            if isinstance(embed_llm, BGEM3Embedding):
                error("Semantic splitter not working with BGE-M3")

            return SemanticSplitterNodeParser(
                buffer_size=splitter.buffer_size,
                breakpoint_percentile_threshold=splitter.breakpoint_percentile_threshold,
                embed_model=embed_llm,
                include_metadata=splitter.include_metadata,
            )
        elif isinstance(splitter, JSONNodeParserConfig):
            return JSONNodeParser(
                include_metadata=splitter.include_metadata,
                include_prev_next_rel=splitter.include_prev_next_rel,
            )
        elif isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            splitter, CodeSplitterConfig
        ):
            return CodeSplitter(
                language=splitter.language,
                chunk_lines=splitter.chunk_lines,
                chunk_lines_overlap=splitter.chunk_lines_overlap,
                max_chars=splitter.max_chars,
            )

    async def __split_documents(
        self,
        models: Models,
        documents: Iterable[Document],
        questions_answered: int | None,
        num_workers: int | None,
        verbose: bool = False,
    ) -> Sequence[BaseNode]:
        self.logger.info("Split documents")

        transformations: list[TransformComponent] = [
            await self.__load_splitter(
                models=models,
                splitter=self.config.splitter,
                verbose=verbose,
            )
        ]

        if models.llm is not None:
            transformations.extend(
                [
                    KeywordExtractor(
                        keywords=5,
                        llm=models.keywords_llm or models.llm,
                        num_workers=self.config.num_workers,
                        show_progress=verbose,
                    ),
                    SummaryExtractor(
                        summaries=["self"],
                        llm=models.metadata_extractor_llm or models.llm,
                        num_workers=self.config.num_workers,
                        show_progress=verbose,
                    ),
                    TitleExtractor(
                        nodes=5,
                        llm=models.metadata_extractor_llm or models.llm,
                        num_workers=self.config.num_workers,
                        show_progress=verbose,
                    ),
                ]
            )
            if questions_answered is not None:
                self.logger.info(
                    f"Generate {questions_answered} questions/chunk"
                )
                transformations.append(
                    QuestionsAnsweredExtractor(
                        questions=questions_answered,
                        llm=models.questions_answered_llm or models.llm,
                        num_workers=self.config.num_workers,
                        show_progress=verbose,
                    )
                )

        # ingest_cache = IngestionCache(
        #     cache=RedisCache.from_host_and_port(host="127.0.0.1", port=6379),
        #     collection="my_test_cache",
        # )

        pipeline: IngestionPipeline = IngestionPipeline(
            transformations=transformations,
            # cache=ingest_cache,
        )
        return await pipeline.arun(
            documents=documents,
            num_workers=num_workers,
            show_progress=verbose,
        )

    async def __populate_vector_store(
        self,
        models: Models,
        nodes: Sequence[BaseNode],
        collect_keywords: bool,
        storage_context: StorageContext,
        verbose: bool = False,
    ) -> tuple[VectorStoreIndex | BGEM3Index, BaseKeywordTableIndex | None]:
        self.logger.info("Populate vector store")

        index_structs = storage_context.index_store.index_structs()
        for struct in index_structs:
            storage_context.index_store.delete_index_struct(
                key=struct.index_id
            )

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
                    nodes,
                    storage_context=storage_context,
                    show_progress=verbose,
                    use_async=True,
                )
            else:
                keyword_index = KeywordTableIndex(
                    nodes,
                    storage_context=storage_context,
                    show_progress=verbose,
                    use_async=True,
                    llm=models.keywords_llm or models.llm,
                )
        else:
            keyword_index = None

        return vector_index, keyword_index

    async def __ingest_documents(
        self,
        storage_context: StorageContext,
        models: Models,
        input_files: list[Path],
        collect_keywords: bool,
        questions_answered: int | None,
        num_workers: int | None,
        verbose: bool = False,
    ) -> tuple[
        VectorStoreIndex | BGEM3Index,
        BaseKeywordTableIndex | None,
    ]:
        documents = await self.__read_documents(
            input_files=input_files,
            num_workers=self.config.num_workers,
            verbose=verbose,
        )

        nodes = await self.__split_documents(
            models,
            documents,
            questions_answered=questions_answered,
            num_workers=num_workers,
            verbose=verbose,
        )

        vector_index, keyword_index = await self.__populate_vector_store(
            models,
            nodes,
            collect_keywords=collect_keywords,
            storage_context=storage_context,
            verbose=verbose,
        )

        vector_index.set_index_id("vector_index")
        if keyword_index is not None:
            keyword_index.set_index_id("keyword_index")

        return vector_index, keyword_index

    async def __save_indices(
        self,
        storage_context: StorageContext,
        vector_index: BGEM3Index | None,
        persist_dir: Path | None,
    ):
        if self.config.db_conn is not None:
            if vector_index is not None:
                self.logger.info("Persist BGE-M3 index to database")
                BGEM3Embedding.persist_to_postgres(
                    self.config.db_conn,
                    vector_index,
                )
        elif persist_dir is not None:
            if vector_index is not None:
                self.logger.info("Persist storage context and BGE-M3 index")
                vector_index.persist(persist_dir=str(persist_dir))
            self.logger.info("Persist storage context to disk")
            storage_context.persist(  # pyright: ignore[reportUnknownMemberType]
                persist_dir=str(persist_dir)
            )

    async def __persist_dir(
        self,
        input_files: list[Path],
        embedding: EmbeddingConfig,
    ) -> Path:
        self.logger.info(f"{len(input_files)} input file(s)")
        fp: str = await self.__determine_fingerprint(
            input_files,
            embedding.model,
            embedding.dimensions,
        )
        self.logger.info(f"Fingerprint = {fp}")
        return cache_dir(fp)

    async def __load_storage_context(
        self,
        persist_dir: Path | None,
    ) -> StorageContext:
        if (
            self.config.db_conn is not None
            and self.config.embedding is not None
        ):
            docstore, index_store, vector_store = await self.__postgres_stores(
                uri=self.config.db_conn,
                embedding_dimensions=self.config.embedding.dimensions,
            )
        elif persist_dir is not None and os.path.isdir(persist_dir):
            docstore = SimpleDocumentStore.from_persist_dir(str(persist_dir))
            index_store = SimpleIndexStore.from_persist_dir(str(persist_dir))
            vector_store = SimpleVectorStore.from_persist_dir(str(persist_dir))
        else:
            docstore = SimpleDocumentStore()
            index_store = SimpleIndexStore()
            vector_store = SimpleVectorStore()

        return StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store,
            vector_store=vector_store,
            persist_dir=(
                str(persist_dir)
                if persist_dir is not None
                else DEFAULT_PERSIST_DIR
            ),
        )

    async def __load_indices(
        self,
        storage_context: StorageContext,
        models: Models,
        persist_dir: Path | None,
        collect_keywords: bool = False,
    ) -> tuple[
        VectorStoreIndex | BGEM3Index | None,
        BaseKeywordTableIndex | None,
    ]:
        vector_index: VectorStoreIndex | BGEM3Index | None = None
        keyword_index: BaseKeywordTableIndex | None = None

        try:
            if self.config.db_conn is not None:
                self.logger.info("Read indices from database")
            else:
                self.logger.info("Read indices from cache")

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
                    self.logger.info("Read BGE-M3 index from cache")
                    vector_index = BGEM3Index.load_from_disk(
                        persist_dir=str(persist_dir),
                        weights_for_different_modes=[0.4, 0.2, 0.4],
                    )
            else:
                indices: list[BaseIndex[IndexDict]] = (
                    load_indices_from_storage(
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
                )
                if collect_keywords:
                    [vi, ki] = indices
                    vector_index = cast(VectorStoreIndex, vi)
                    keyword_index = cast(BaseKeywordTableIndex, ki)
                else:
                    [vi] = indices
                    vector_index = cast(VectorStoreIndex, vi)

        except ValueError:
            self.logger.info("Failed to read indices")

        return vector_index, keyword_index

    async def __ingest_files(
        self,
        models: Models,
        input_files: list[Path] | None,
        collect_keywords: bool,
        questions_answered: int | None,
        num_workers: int | None,
        verbose: bool = False,
    ) -> tuple[
        VectorStoreIndex | BGEM3Index | None,
        BaseKeywordTableIndex | None,
    ]:
        persist_dir: Path | None = None
        persisted = False

        if input_files is None:
            self.logger.info("No input files")
        elif self.config.embedding is None:
            error("Cannot ingest files without an embedding model")
        else:
            persist_dir = await self.__persist_dir(
                input_files,
                self.config.embedding,
            )
            persisted = os.path.isdir(persist_dir)

        storage_context = await self.__load_storage_context(
            persist_dir=persist_dir,
        )

        if self.config.db_conn is not None or persisted:
            vector_index, keyword_index = await self.__load_indices(
                storage_context=storage_context,
                models=models,
                persist_dir=persist_dir,
                collect_keywords=collect_keywords,
            )
        else:
            vector_index = None
            keyword_index = None

        if input_files is not None and not persisted:
            vector_index, keyword_index = await self.__ingest_documents(
                storage_context=storage_context,
                models=models,
                input_files=input_files,
                collect_keywords=collect_keywords,
                questions_answered=questions_answered,
                num_workers=num_workers,
                verbose=verbose,
            )
            await self.__save_indices(
                storage_context,
                vector_index if isinstance(vector_index, BGEM3Index) else None,
                persist_dir=persist_dir,
            )

        return (vector_index, keyword_index)

    async def load_retriever(
        self,
        models: Models,
        input_files: list[Path] | None,
        verbose: bool = False,
    ) -> BaseRetriever | None:
        vector_index, keyword_index = await self.__ingest_files(
            models=models,
            input_files=input_files,
            collect_keywords=self.config.collect_keywords,
            questions_answered=self.config.questions_answered,
            num_workers=self.config.num_workers,
            verbose=verbose,
        )

        if vector_index is not None:
            if self.config.db_conn is not None and self.config.hybrid_search:
                self.logger.info("Create fusion vector retriever")
                vector_retriever = vector_index.as_retriever(
                    vector_store_query_mode="default",
                    similarity_top_k=5,
                    verbose=verbose,
                )
                text_retriever = vector_index.as_retriever(
                    vector_store_query_mode="sparse",
                    similarity_top_k=5,  # interchangeable with sparse_top_k
                    verbose=verbose,
                )
                vector_retriever = QueryFusionRetriever(
                    [vector_retriever, text_retriever],
                    similarity_top_k=self.config.top_k,
                    num_queries=1,  # set this to 1 to disable query generation
                    mode=FUSION_MODES.RELATIVE_SCORE,
                    llm=models.llm,
                    use_async=True,
                    verbose=verbose,
                )
            else:
                self.logger.info("Create vector retriever")
                vector_retriever = vector_index.as_retriever(
                    similarity_top_k=self.config.top_k,
                    verbose=verbose,
                )
        else:
            vector_retriever = None

        if keyword_index is not None:
            self.logger.info("Create keyword retriever")
            keyword_retriever = keyword_index.as_retriever(
                verbose=verbose,
            )
        else:
            keyword_retriever = None

        if vector_retriever is not None:
            if keyword_retriever is not None:
                self.logger.info("Create aggregate custom retriever")
                retriever = CustomRetriever(
                    vector_retriever=vector_retriever,
                    keyword_retriever=keyword_retriever,
                    verbose=verbose,
                )
            else:
                retriever = vector_retriever
        else:
            retriever = keyword_retriever

        return retriever

    async def retrieve_nodes(
        self, retriever: BaseRetriever, text: str
    ) -> list[dict[str, Any]]:
        self.logger.info("Retrieve nodes from vector index")
        nodes = await retriever.aretrieve(text)
        self.logger.info(f"{len(nodes)} nodes found in vector index")
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

        self.logger.info("Query with retriever query engine")
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
            _guideline_evaluator = GuidelineEvaluator(
                llm=models.evaluator_llm or models.llm,
                guidelines=DEFAULT_GUIDELINES
                + "\nThe response should not be overly long.\n"
                + "The response should try to summarize where possible.\n",
            )
            if source_retries:
                self.logger.info("Add retry source query engine")
                query_engine = RetrySourceQueryEngine(
                    query_engine,
                    evaluator=relevancy_evaluator,
                    llm=models.llm,
                )
            else:
                self.logger.info("Add retry query engine")
                query_engine = RetryQueryEngine(
                    query_engine, evaluator=relevancy_evaluator
                )

        self.logger.info("Submit query to LLM")
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
        chat_history: list[ChatMessage] | None = None,
        system_prompt: str | None = None,
        summarize_chat: bool = False,
        streaming: bool = False,
    ) -> StreamingAgentChatResponse | AgentChatResponse | NoReturn:
        if models.llm is None:
            error("There is no LLM configured to chat with")

        if chat_store is None and chat_history is not None:
            await self.reset_chat()
            chat_store = SimpleChatStore()
            chat_store.set_messages(key=user, messages=chat_history)

        if self.chat_memory is None and chat_store is not None:
            if summarize_chat:
                self.chat_memory = ChatSummaryMemoryBuffer.from_defaults(  # pyright: ignore[reportUnknownMemberType]
                    token_limit=token_limit,
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
                    system_prompt=system_prompt,
                )
            else:
                self.chat_engine = ContextChatEngine.from_defaults(
                    retriever=retriever,
                    llm=models.llm,
                    memory=self.chat_memory,
                    system_prompt=system_prompt,
                )

        self.logger.info("Submit chat to LLM")
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
    pplx_api_key = (
        "your-perplexity-api-key"  # Replace with your actual API key
    )

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
    config_path: Path,
    input_from: str | None,
    verbose: bool = False,
) -> tuple[RAGWorkflow, Models, BaseRetriever | None]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = await RAGWorkflow.load_config(config_path)

    if input_from is not None:
        input_files = read_files(input_from, config.recursive)
    else:
        input_files = awaitable_none()

    rag = RAGWorkflow(config, logging.getLogger("rag"))

    models = await rag.initialize(
        verbose=verbose,
    )

    if config.embedding is not None:
        retriever = await rag.load_retriever(
            models=models,
            input_files=await input_files,
            verbose=verbose,
        )
    else:
        retriever = None

    return (rag, models, retriever)


def rebuild_postgres_db(db_name: str):
    connection_string = "postgresql://postgres:password@localhost:5432"
    with psycopg2.connect(connection_string) as conn:
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {db_name}")
            c.execute(f"CREATE DATABASE {db_name}")

    connection_string2 = (
        "postgresql://postgres:password@localhost:5432/{db_name}"
    )
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
