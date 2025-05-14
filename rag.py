# pyright: reportMissingTypeStubs=false
# pyright: reportExplicitAny=false
# pyright: reportAny=false

import base64
import hashlib
import logging
import os
import sys
import psycopg2
import llama_cpp

from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from dataclass_wizard import JSONWizard, YAMLWizard
from functools import cache
from pathlib import Path
from pydantic import BaseModel
from urllib.parse import urlparse
from orgparse.node import OrgNode
from xdg_base_dirs import xdg_cache_home
from typing import (
    Any,
    Literal,
    NoReturn,
    TypeAlias,
    cast,
    final,
    no_type_check,
    override,
)

from llama_index.core import (
    KeywordTableIndex,
    PromptTemplate,
    QueryBundle,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,  # pyright: ignore[reportUnknownVariableType]
    load_indices_from_storage,  # pyright: ignore[reportUnknownVariableType]
)
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.query_engine.custom import STR_OR_RESPONSE_TYPE
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.chat_engine import (
    CondensePlusContextChatEngine,
    SimpleChatEngine,
)
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    BaseChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.embeddings import BaseEmbedding

# from llama_index.core.evaluation import GuidelineEvaluator, RelevancyEvaluator
# from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
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
    BaseQueryEngine,
    CitationQueryEngine,
    CustomQueryEngine,
    RetrieverQueryEngine,
    # RetryQueryEngine,
    # RetrySourceQueryEngine,
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
)
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import (
    BaseNode,
    Document,
    Node,
    NodeWithScore,
    QueryType,
    TransformComponent,
)
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.storage_context import DEFAULT_PERSIST_DIR
from llama_index.core.vector_stores.simple import SimpleVectorStore

# from llama_index.core.tools import FunctionTool

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.indices.managed.bge_m3 import BGEM3Index
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai.base import DEFAULT_OPENAI_MODEL
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.perplexity import Perplexity
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.index_store.postgres import PostgresIndexStore
from llama_index.vector_stores.postgres import PGVectorStore


# Utility functions


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


def read_files(
    read_from: str,
    recursive: bool = False,
) -> list[Path] | NoReturn:
    if read_from == "-":
        input_files = [Path(line.strip()) for line in sys.stdin if line.strip()]
        if not input_files:
            error("No filenames provided on standard input")
        return input_files
    elif os.path.isdir(read_from):
        return list_files(Path(read_from), recursive)
    elif os.path.isfile(read_from):
        return [Path(read_from)]
    else:
        error(f"Input path is unrecognized or non-existent: {read_from}")


def convert_str(read_from: str | None) -> str | None:
    if read_from is None:
        return read_from
    elif read_from == "-":
        s = sys.stdin.read()
        if not s:
            error("No input provided on standard input")
        return s
    elif os.path.isfile(read_from):
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


@final
class GlobalJSONMeta(JSONWizard.Meta):
    tag_key = "type"
    auto_assign_tags = True
    # v1_debug = logging.INFO


@dataclass
class LLMConfig(YAMLWizard):
    provider: str = "OpenAI"
    model: str = DEFAULT_OPENAI_MODEL
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
class LlamaCPPEmbeddingConfig(EmbeddingConfig):
    model_path: Path = field(default_factory=Path)
    n_gpu_layers: int = 0
    split_mode: int = llama_cpp.LLAMA_SPLIT_MODE_LAYER
    main_gpu: int = 0
    tensor_split: list[float] | None = None
    rpc_servers: str | None = None
    vocab_only: bool = False
    use_mmap: bool = True
    use_mlock: bool = False
    kv_overrides: dict[str, bool | int | float | str] | None = None
    # Context Params
    seed: int = llama_cpp.LLAMA_DEFAULT_SEED
    n_ctx: int = 512
    n_batch: int = 512
    n_ubatch: int = 512
    n_threads: int | None = None
    n_threads_batch: int | None = None
    rope_scaling_type: int | None = llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    pooling_type: int = llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED
    rope_freq_base: float = 0.0
    rope_freq_scale: float = 0.0
    yarn_ext_factor: float = -1.0
    yarn_attn_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_orig_ctx: int = 0
    logits_all: bool = False
    embedding: bool = False
    offload_kqv: bool = True
    flash_attn: bool = False
    # Sampling Params
    no_perf: bool = False
    last_n_tokens_size: int = 64
    # LoRA Params
    lora_base: str | None = None
    lora_scale: float = 1.0
    lora_path: str | None = None
    # Backend Params
    numa: bool | int = False
    # Chat Format Params
    chat_format: str | None = None
    chat_handler: llama_cpp.llama_chat_format.LlamaChatCompletionHandler | None = None
    # Speculative Decoding
    draft_model: llama_cpp.LlamaDraftModel | None = None
    # Tokenizer Override
    tokenizer: llama_cpp.BaseLlamaTokenizer | None = None
    # KV cache quantization
    type_k: int | None = None
    type_v: int | None = None
    # Misc
    spm_infill: bool = False


@dataclass
class KeywordsConfig(YAMLWizard):
    collect: bool = False
    llm: LLMConfig | None = None


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


SplitterConfig: TypeAlias = (
    SentenceSplitterConfig
    | SentenceWindowSplitterConfig
    | SemanticSplitterConfig
    | JSONNodeParserConfig
    | CodeSplitterConfig
)


@dataclass
class KeywordExtractorConfig(YAMLWizard):
    llm: LLMConfig = field(default_factory=LLMConfig)
    keywords: int = 5


@dataclass
class SummaryExtractorConfig(YAMLWizard):
    llm: LLMConfig = field(default_factory=LLMConfig)
    # ["self"]
    summaries: list[str] = field(default_factory=list)


@dataclass
class TitleExtractorConfig(YAMLWizard):
    llm: LLMConfig = field(default_factory=LLMConfig)
    nodes: int = 5


@dataclass
class QuestionsAnsweredExtractorConfig(YAMLWizard):
    llm: LLMConfig = field(default_factory=LLMConfig)
    questions: int = 1


ExtractorConfig: TypeAlias = (
    KeywordExtractorConfig
    | SummaryExtractorConfig
    | TitleExtractorConfig
    | QuestionsAnsweredExtractorConfig
)


@dataclass
class SimpleChatEngineConfig(YAMLWizard):
    pass


@dataclass
class ContextChatEngineConfig(YAMLWizard):
    pass


@dataclass
class CondensePlusContextChatEngineConfig(YAMLWizard):
    skip_condense: bool = False


ChatEngineConfig: TypeAlias = (
    SimpleChatEngineConfig
    | ContextChatEngineConfig
    | CondensePlusContextChatEngineConfig
)


@dataclass
class VectorStoreConfig(YAMLWizard):
    pass


@dataclass
class PostgresVectorStoreConfig(YAMLWizard):
    connection: str
    hybrid_search: bool = False
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    hnsw_dist_method: str = "vector_cosine_ops"


@dataclass
class RetrievalConfig(YAMLWizard):
    @final
    class _(JSONWizard.Meta):
        # v1 = True  # Enable v1 opt-in
        # v1_unsafe_parse_dataclass_in_union = True
        tag_key = "type"
        auto_assign_tags = True

    embedding: EmbeddingConfig | None = None
    keywords: KeywordsConfig = field(default_factory=KeywordsConfig)
    splitter: SplitterConfig = field(default_factory=SentenceSplitterConfig)
    extractors: list[ExtractorConfig] = field(default_factory=list)
    vector_store: VectorStoreConfig | None = None
    top_k: int = 3


@dataclass
class QueryConfig(YAMLWizard):
    llm: LLMConfig = field(default_factory=LLMConfig)
    retries: bool = False
    source_retries: bool = False
    show_citations: bool = False
    evaluator_llm: LLMConfig | None = None


@dataclass
class ChatConfig(YAMLWizard):
    @final
    class _(JSONWizard.Meta):
        # v1 = True  # Enable v1 opt-in
        # v1_unsafe_parse_dataclass_in_union = True
        tag_key = "type"
        auto_assign_tags = True

    llm: LLMConfig = field(default_factory=LLMConfig)
    engine: ChatEngineConfig = field(default_factory=SimpleChatEngineConfig)
    default_user: str = "user"
    summarize: bool = False
    keep_history: bool = False


@dataclass
class Config(YAMLWizard):
    query: QueryConfig = field(default_factory=QueryConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


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


class SimpleQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    llm: LLM
    response_synthesizer: BaseSynthesizer
    qa_prompt: PromptTemplate = field(
        default=PromptTemplate("Query: {query_str}\nAnswer: ")
    )

    def __init__(self, **kwargs: Any):
        self.response_synthesizer = kwargs[
            "response_synthesizer"
        ] or get_response_synthesizer(response_mode=ResponseMode.COMPACT)
        super().__init__(**kwargs)

    @override
    def custom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
        response = self.llm.complete(self.qa_prompt.format(query_str=query_str))
        return str(response)


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


EmbeddingType: TypeAlias = BaseEmbedding | BGEM3Embedding
FunctionsType: TypeAlias = list[dict[str, Any | None]] | None


class Message(BaseModel):
    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    "https://platform.openai.com/docs/api-reference/chat/create"
    messages: list[Message]
    model: str
    # audio
    frequency_penalty: float | None = 0
    function_call: str | dict[str, str | None] | None = None
    # ^ instead use tool_choice
    functions: FunctionsType = None
    # ^ instead use tools
    # logit_bias
    # logprobs
    # max_completion_tokens
    max_tokens: int | None = None
    # ^ instead use max_completion_tokens
    # metadata
    # modalities
    n: int | None = 1
    # parallel_tool_calls
    # prediction
    presence_penalty: float | None = 0
    # reasoning_effort: str | None = None
    # response_format
    # seed
    # service_tier
    # stop
    # store
    stream: bool | None = False
    # stream_options
    temperature: float | None = 0.7
    # tool_choice
    # tools
    # top_logprobs
    top_p: float | None = 1.0
    user: str | None = None
    # web_search_options


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    max_tokens: int | None = 16
    presence_penalty: float | None = 0
    frequency_penalty: float | None = 0
    user: str | None = None


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    user: str | None = None


@dataclass
class RAGWorkflow:
    config: Config

    @classmethod
    def load_config(cls, path: Path) -> Config:
        if os.path.isfile(path):
            cfg = Config.from_yaml_file(  # pyright: ignore[reportUnknownMemberType]
                str(path)
            )
            if isinstance(cfg, Config):
                if cfg.retrieval.embedding is not None:
                    if cfg.retrieval.embedding.provider == "BGEM3":
                        cfg.retrieval.embedding.model = "BAAI/bge-m3"
                return cfg
            else:
                error("Config file should define a single Config object")
        else:
            error(f"Cannot read config file {path}")

    def __postgres_stores(
        self,
        config: PostgresVectorStoreConfig,
        embedding_dimensions: int,
    ) -> tuple[PostgresDocumentStore, PostgresIndexStore, PGVectorStore]:
        docstore: PostgresDocumentStore = PostgresDocumentStore.from_uri(
            uri=config.connection,
            table_name="docstore",
        )
        index_store: PostgresIndexStore = PostgresIndexStore.from_uri(
            uri=config.connection,
            table_name="indexstore",
        )

        details: PostgresDetails = PostgresDetails(config.connection)

        vector_store: PGVectorStore = PGVectorStore.from_params(
            connection_string=config.connection,
            database=details.database,
            host=details.host,
            password=details.password,
            port=str(details.port),
            user=details.user,
            table_name="vectorstore",
            embed_dim=embedding_dimensions,
            hybrid_search=config.hybrid_search,
            hnsw_kwargs={
                "hnsw_m": config.hnsw_m,
                "hnsw_ef_construction": config.hnsw_ef_construction,
                "hnsw_ef_search": config.hnsw_ef_search,
                "hnsw_dist_method": config.hnsw_dist_method,
            },
        )
        return docstore, index_store, vector_store

    def __load_embedding(
        self,
        embed_config: EmbeddingConfig,
        verbose: bool = False,
    ) -> EmbeddingType:
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
                parallel_process=False,
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

    @classmethod
    def __load_llm(
        cls,
        llm_config: LLMConfig,
        verbose: bool = False,
    ) -> LLM:
        if llm_config.provider == "Ollama":
            return Ollama(
                model=llm_config.model,
                base_url=llm_config.base_url or "http://localhost:11434",
                temperature=llm_config.temperature,
                context_window=llm_config.context_window,
                max_tokens=llm_config.max_tokens,
                system_prompt=convert_str(llm_config.system_prompt),
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
                system_prompt=convert_str(llm_config.system_prompt),
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
                system_prompt=convert_str(llm_config.system_prompt),
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
                system_prompt=convert_str(llm_config.system_prompt),
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
                system_prompt=convert_str(llm_config.system_prompt),
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
                system_prompt=convert_str(llm_config.system_prompt),
                is_chat_model=True,
            )
        elif llm_config.provider == "LMStudio":
            return LMStudio(
                model_name=llm_config.model,
                base_url=llm_config.base_url or "http://localhost:1234/v1",
                temperature=llm_config.temperature,
                context_window=llm_config.context_window,
                timeout=llm_config.timeout,
                system_prompt=convert_str(llm_config.system_prompt),
                is_chat_model=True,
                # request_timeout,
                # num_output,
            )
        else:
            error(f"LLM model not recognized: {llm_config.model}")

    @classmethod
    def realize_llm(
        cls,
        llm_config: LLMConfig | None,
        verbose: bool = False,
    ) -> LLM | NoReturn:
        llm = (
            cls.__load_llm(
                llm_config=llm_config,
                verbose=verbose,
            )
            if llm_config is not None
            else None
        )
        if llm is None:
            error(f"Failed to start LLM: {llm_config}")
        else:
            return llm

    def __determine_fingerprint(
        self,
        input_files: list[Path],
        embed_model: str,
        embed_dim: int,
    ) -> str:
        fingerprint = [
            collection_hash(input_files),
            hashlib.sha512(embed_model.encode("utf-8")).hexdigest(),
            hashlib.sha512(str(embed_dim).encode("utf-8")).hexdigest(),
        ]
        final_hash = "\n".join(fingerprint).encode("utf-8")
        final_base64 = base64.b64encode(final_hash).decode("utf-8")
        return final_base64[0:32]

    def __read_documents(
        self,
        input_files: list[Path],
        num_workers: int | None = None,
        recursive: bool = False,
        verbose: bool = False,
    ) -> Iterable[Document]:
        file_extractor: dict[str, BaseReader] = {
            ".org": OrgReader(),
        }
        return SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=file_extractor,
            recursive=recursive,
        ).load_data(num_workers=num_workers, show_progress=verbose)

    def __load_splitter(
        self,
        splitter: SplitterConfig,
        verbose: bool = False,
    ) -> TransformComponent | NoReturn:
        match splitter:
            case SentenceSplitterConfig():
                return SentenceSplitter(
                    chunk_size=splitter.chunk_size,
                    chunk_overlap=splitter.chunk_overlap,
                    include_metadata=splitter.include_metadata,
                )
            case SentenceWindowSplitterConfig():
                return SentenceWindowNodeParser.from_defaults(
                    window_size=splitter.window_size,
                    window_metadata_key=splitter.window_metadata_key,
                    original_text_metadata_key=splitter.original_text_metadata_key,
                )
            case SemanticSplitterConfig():
                if splitter.embedding is not None:
                    embed_llm = self.__load_embedding(
                        splitter.embedding,
                        verbose=verbose,
                    )
                else:
                    error("Semantic splitter needs an embedding model")

                if isinstance(embed_llm, BGEM3Embedding):
                    error("Semantic splitter not working with BGE-M3")

                return SemanticSplitterNodeParser(
                    buffer_size=splitter.buffer_size,
                    breakpoint_percentile_threshold=splitter.breakpoint_percentile_threshold,
                    embed_model=embed_llm,
                    include_metadata=splitter.include_metadata,
                )
            case JSONNodeParserConfig():
                return JSONNodeParser(
                    include_metadata=splitter.include_metadata,
                    include_prev_next_rel=splitter.include_prev_next_rel,
                )
            case CodeSplitterConfig():
                return CodeSplitter(
                    language=splitter.language,
                    chunk_lines=splitter.chunk_lines,
                    chunk_lines_overlap=splitter.chunk_lines_overlap,
                    max_chars=splitter.max_chars,
                )

    def __load_extractor(
        self,
        config: ExtractorConfig,
        verbose: bool = False,
    ) -> TransformComponent | None:
        match config:
            case KeywordExtractorConfig():
                llm = self.realize_llm(config.llm, verbose=verbose)
                return KeywordExtractor(
                    keywords=config.keywords,
                    llm=llm,
                    show_progress=verbose,
                )
            case SummaryExtractorConfig():
                llm = self.realize_llm(config.llm, verbose=verbose)
                return SummaryExtractor(
                    summaries=config.summaries,
                    llm=llm,
                    show_progress=verbose,
                )
            case TitleExtractorConfig():
                llm = self.realize_llm(config.llm, verbose=verbose)
                return TitleExtractor(
                    nodes=config.nodes,
                    llm=llm,
                    show_progress=verbose,
                )
            case QuestionsAnsweredExtractorConfig():
                llm = self.realize_llm(config.llm, verbose=verbose)
                return QuestionsAnsweredExtractor(
                    questions=config.questions,
                    llm=llm,
                    show_progress=verbose,
                )

    def __split_documents(
        self,
        documents: Iterable[Document],
        verbose: bool = False,
    ) -> Sequence[BaseNode]:
        transformations: list[TransformComponent] = [
            self.__load_splitter(
                splitter=self.config.retrieval.splitter,
                verbose=verbose,
            )
        ]

        extractors = [
            self.__load_extractor(
                entry,
                verbose=verbose,
            )
            for entry in self.config.retrieval.extractors
        ]

        transformations.extend([x for x in extractors if x is not None])

        # ingest_cache = IngestionCache(
        #     cache=RedisCache.from_host_and_port(host="127.0.0.1", port=6379),
        #     collection="my_test_cache",
        # )

        pipeline: IngestionPipeline = IngestionPipeline(
            transformations=transformations,
            # cache=ingest_cache,
        )
        return pipeline.run(
            documents=documents,
            show_progress=verbose,
        )

    def __populate_vector_store(
        self,
        embed_llm: EmbeddingType,
        nodes: Sequence[BaseNode],
        storage_context: StorageContext,
        verbose: bool = False,
    ) -> tuple[VectorStoreIndex | BGEM3Index, BaseKeywordTableIndex | None]:
        index_structs = storage_context.index_store.index_structs()
        for struct in index_structs:
            storage_context.index_store.delete_index_struct(key=struct.index_id)

        match embed_llm:
            case BGEM3Embedding():
                vector_index = BGEM3Index(
                    nodes,
                    storage_context=storage_context,
                    weights_for_different_modes=[0.4, 0.2, 0.4],
                    show_progress=verbose,
                )
            case BaseEmbedding():
                vector_index = VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    embed_model=embed_llm,
                    show_progress=verbose,
                )

        if self.config.retrieval.keywords.collect:
            if self.config.retrieval.keywords.llm is None:
                keyword_index = SimpleKeywordTableIndex(
                    nodes,
                    storage_context=storage_context,
                    show_progress=verbose,
                )
            else:
                llm = self.realize_llm(self.config.retrieval.keywords.llm)
                keyword_index = KeywordTableIndex(
                    nodes,
                    storage_context=storage_context,
                    show_progress=verbose,
                    llm=llm,
                )
        else:
            keyword_index = None

        return vector_index, keyword_index

    def __ingest_documents(
        self,
        embed_llm: EmbeddingType,
        storage_context: StorageContext,
        input_files: list[Path],
        num_workers: int | None = None,
        verbose: bool = False,
    ) -> tuple[
        VectorStoreIndex | BGEM3Index,
        BaseKeywordTableIndex | None,
    ]:
        documents = self.__read_documents(
            num_workers=num_workers,
            input_files=input_files,
            verbose=verbose,
        )

        nodes = self.__split_documents(
            documents,
            verbose=verbose,
        )

        vector_index, keyword_index = self.__populate_vector_store(
            embed_llm=embed_llm,
            nodes=nodes,
            storage_context=storage_context,
            verbose=verbose,
        )

        vector_index.set_index_id("vector_index")
        if keyword_index is not None:
            keyword_index.set_index_id("keyword_index")

        return vector_index, keyword_index

    def __save_indices(
        self,
        storage_context: StorageContext,
        vector_index: BGEM3Index | None,
        persist_dir: Path | None,
    ):
        if self.config.retrieval.vector_store is not None:
            if vector_index is not None and isinstance(
                self.config.retrieval.vector_store, PostgresVectorStoreConfig
            ):
                BGEM3Embedding.persist_to_postgres(
                    self.config.retrieval.vector_store.connection,
                    vector_index,
                )
        elif persist_dir is not None:
            if vector_index is not None:
                vector_index.persist(persist_dir=str(persist_dir))
            storage_context.persist(  # pyright: ignore[reportUnknownMemberType]
                persist_dir=str(persist_dir)
            )

    def __persist_dir(
        self,
        input_files: list[Path],
        embedding: EmbeddingConfig,
    ) -> Path:
        fp: str = self.__determine_fingerprint(
            input_files,
            embedding.model,
            embedding.dimensions,
        )
        return cache_dir(fp)

    def __load_storage_context(
        self,
        persist_dir: Path | None,
    ) -> StorageContext:
        if (
            self.config.retrieval.vector_store is not None
            and isinstance(
                self.config.retrieval.vector_store, PostgresVectorStoreConfig
            )
            and self.config.retrieval.embedding is not None
        ):
            docstore, index_store, vector_store = self.__postgres_stores(
                config=self.config.retrieval.vector_store,
                embedding_dimensions=self.config.retrieval.embedding.dimensions,
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
                str(persist_dir) if persist_dir is not None else DEFAULT_PERSIST_DIR
            ),
        )

    def __load_indices(
        self,
        embed_llm: EmbeddingType,
        storage_context: StorageContext,
        persist_dir: Path | None,
    ) -> tuple[
        VectorStoreIndex | BGEM3Index | None,
        BaseKeywordTableIndex | None,
    ]:
        keyword_index = None

        match embed_llm:
            case BGEM3Embedding():
                if self.config.retrieval.vector_store is not None:
                    error("BGE-M3 not current compatible with databases")
                    # logger.info("Read BGE-M3 index from database")
                    # self.vector_index = BGEM3Embedding.load_from_postgres(
                    #     uri=self.config.db_conn,
                    #     storage_context=self.storage_context,
                    #     weights_for_different_modes=[0.4, 0.2, 0.4],
                    # )
                else:
                    vector_index = BGEM3Index.load_from_disk(
                        persist_dir=str(persist_dir),
                        weights_for_different_modes=[0.4, 0.2, 0.4],
                    )
            case BaseEmbedding():
                indices: list[BaseIndex[IndexDict]] = load_indices_from_storage(
                    storage_context=storage_context,
                    embed_model=(
                        embed_llm if not isinstance(embed_llm, BGEM3Embedding) else None
                    ),
                    # This doesn't actually need an llm, but it tries to
                    # load one if it's None
                    llm=embed_llm,
                    index_ids=(
                        ["vector_index", "keyword_index"]
                        if self.config.retrieval.keywords.collect
                        else ["vector_index"]
                    ),
                )
                if self.config.retrieval.keywords.collect:
                    [vi, ki] = indices
                    vector_index = cast(VectorStoreIndex, vi)
                    keyword_index = cast(BaseKeywordTableIndex, ki)
                else:
                    [vi] = indices
                    vector_index = cast(VectorStoreIndex, vi)

        return vector_index, keyword_index

    def __ingest_files(
        self,
        input_files: list[Path] | None,
        num_workers: int | None = None,
        verbose: bool = False,
    ) -> tuple[
        VectorStoreIndex | BGEM3Index | None,
        BaseKeywordTableIndex | None,
    ]:
        persist_dir: Path | None = None
        persisted = False

        if input_files is None:
            pass
        elif self.config.retrieval.embedding is None:
            error("Cannot ingest files without an embedding model")
        else:
            persist_dir = self.__persist_dir(
                input_files,
                self.config.retrieval.embedding,
            )
            persisted = os.path.isdir(persist_dir)

        storage_context = self.__load_storage_context(
            persist_dir=persist_dir,
        )

        if self.config.retrieval.embedding is not None:
            embed_llm = self.__load_embedding(
                embed_config=self.config.retrieval.embedding,
                verbose=verbose,
            )
        else:
            error("File ingestion requires an embedding model")

        vector_index = None
        keyword_index = None

        if self.config.retrieval.vector_store is not None or persisted:
            try:
                vector_index, keyword_index = self.__load_indices(
                    embed_llm=embed_llm,
                    storage_context=storage_context,
                    persist_dir=persist_dir,
                )
            except ValueError:
                pass

        if input_files is not None and not persisted:
            vector_index, keyword_index = self.__ingest_documents(
                num_workers=num_workers,
                embed_llm=embed_llm,
                storage_context=storage_context,
                input_files=input_files,
                verbose=verbose,
            )
            self.__save_indices(
                storage_context,
                vector_index if isinstance(vector_index, BGEM3Index) else None,
                persist_dir=persist_dir,
            )

        return (vector_index, keyword_index)

    def load_retriever(
        self,
        input_files: list[Path] | None,
        num_workers: int | None = None,
        verbose: bool = False,
    ) -> BaseRetriever | None:
        vector_index, keyword_index = self.__ingest_files(
            num_workers=num_workers,
            input_files=input_files,
            verbose=verbose,
        )

        if vector_index is not None:
            if (
                self.config.retrieval.vector_store is not None
                and isinstance(
                    self.config.retrieval.vector_store, PostgresVectorStoreConfig
                )
                and self.config.retrieval.vector_store.hybrid_search
            ):
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
                    similarity_top_k=self.config.retrieval.top_k,
                    num_queries=1,  # set this to 1 to disable query generation
                    mode=FUSION_MODES.RELATIVE_SCORE,
                    llm=None,  # jww (2025-05-12): FIXME
                    verbose=verbose,
                )
            else:
                vector_retriever = vector_index.as_retriever(
                    similarity_top_k=self.config.retrieval.top_k,
                    verbose=verbose,
                )
        else:
            vector_retriever = None

        if keyword_index is not None:
            keyword_retriever = keyword_index.as_retriever(
                verbose=verbose,
            )
        else:
            keyword_retriever = None

        if vector_retriever is not None:
            if keyword_retriever is not None:
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

    def retrieve_nodes(
        self, retriever: BaseRetriever, text: str
    ) -> list[dict[str, Any]]:
        nodes = retriever.retrieve(text)
        return [
            {
                "text": node.text,
                "metadata": node.metadata,
            }
            for node in nodes
        ]


@dataclass
class QueryState:
    query_engine: BaseQueryEngine

    def __init__(
        self,
        config: QueryConfig,
        llm: LLM,
        retriever: BaseRetriever | None = None,
        streaming: bool = False,
        verbose: bool = False,
    ):
        # jww (2025-05-13): Add MultiStepQueryEngine
        if retriever is not None and config.show_citations:
            self.query_engine = CitationQueryEngine(
                retriever=retriever,
                llm=llm,
                # here we can control how granular citation sources are, the
                # default is 512
                # citation_chunk_size=self.config.embedding.chunk_size,
                # citation_chunk_overlap=self.config.embedding.chunk_overlap,
            )
        elif retriever is not None:
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                llm=llm,
                streaming=streaming,
                response_mode=ResponseMode.REFINE,
                # response_mode=ResponseMode.COMPACT,
                # response_mode=ResponseMode.SIMPLE_SUMMARIZE,
                # response_mode=ResponseMode.TREE_SUMMARIZE,
                # response_mode=ResponseMode.GENERATION, # ignore context
                # response_mode=ResponseMode.NO_TEXT, # only context
                # response_mode=ResponseMode.CONTEXT_ONLY,
                # response_mode=ResponseMode.ACCUMULATE,
                # response_mode=ResponseMode.COMPACT_ACCUMULATE,
                verbose=verbose,
            )

            # jww (2025-05-12): restore
            # if retries or source_retries:
            #     relevancy_evaluator = RelevancyEvaluator(
            #         llm=models.evaluator_llm or models.llm,
            #     )
            #     _guideline_evaluator = GuidelineEvaluator(
            #         llm=models.evaluator_llm or models.llm,
            #         guidelines=DEFAULT_GUIDELINES
            #         + "\nThe response should not be overly long.\n"
            #         + "The response should try to summarize where possible.\n",
            #     )
            #     if source_retries:
            #         self.logger.info("Add retry source query engine")
            #         query_engine = RetrySourceQueryEngine(
            #             query_engine,
            #             evaluator=relevancy_evaluator,
            #             llm=models.llm,
            #         )
            #     else:
            #         self.logger.info("Add retry query engine")
            #         query_engine = RetryQueryEngine(
            #             query_engine,
            #             evaluator=relevancy_evaluator,
            #         )
        else:
            self.query_engine = SimpleQueryEngine(llm=llm)

    def query(self, query: QueryType) -> RESPONSE_TYPE:
        return self.query_engine.query(query)

    async def aquery(self, query: QueryType) -> RESPONSE_TYPE:
        return await self.query_engine.aquery(query)


@dataclass
class ChatState:
    "Chat with the LLM, possibly in the context of a document collection."
    chat_engine: BaseChatEngine

    def __init__(
        self,
        config: ChatConfig,
        llm: LLM,
        user: str,
        chat_store: SimpleChatStore | None = None,
        chat_history: list[ChatMessage] | None = None,
        retriever: BaseRetriever | None = None,
        token_limit: int = 1500,
        system_prompt: str | None = None,
        verbose: bool = False,
    ):
        chat_store = SimpleChatStore()
        if chat_history is not None:
            chat_store.set_messages(key=user, messages=chat_history)

        if config.summarize:
            chat_memory = ChatSummaryMemoryBuffer.from_defaults(  # pyright: ignore[reportUnknownMemberType]
                token_limit=token_limit,
                summarize_prompt=(
                    "The following is a conversation between the user and assistant. "
                    "Write a concise summary about the contents of this conversation."
                ),
                chat_store=chat_store,
                chat_story_key=user,
                llm=llm,
            )
        else:
            chat_memory = ChatMemoryBuffer.from_defaults(  # pyright: ignore[reportUnknownMemberType]
                token_limit=token_limit,
                chat_store=chat_store,
                chat_store_key=user,
                llm=llm,
            )

        match config.engine:
            case ContextChatEngineConfig():
                if retriever is None:
                    error("ContextChatEngine requires a retriever")
                chat_engine = ContextChatEngine.from_defaults(
                    retriever=retriever,
                    llm=llm,
                    memory=chat_memory,
                    system_prompt=system_prompt,
                    verbose=verbose,
                )
            case CondensePlusContextChatEngineConfig():
                if retriever is None:
                    error("ContextChatEngine requires a retriever")
                chat_engine = CondensePlusContextChatEngine.from_defaults(
                    retriever=retriever,
                    llm=llm,
                    memory=chat_memory,
                    system_prompt=system_prompt,
                    skip_condense=config.engine.skip_condense,
                    verbose=verbose,
                )
            case SimpleChatEngineConfig():
                chat_engine = SimpleChatEngine.from_defaults(
                    llm=llm,
                    memory=chat_memory,
                    system_prompt=system_prompt,
                    verbose=verbose,
                )

        self.chat_engine = chat_engine

    def chat(
        self,
        query: str,
        streaming: bool = False,
    ) -> StreamingAgentChatResponse | AGENT_CHAT_RESPONSE_TYPE:
        if streaming:
            return self.chat_engine.stream_chat(query)
        else:
            return self.chat_engine.chat(query)

    async def achat(
        self,
        query: str,
        streaming: bool = False,
    ) -> StreamingAgentChatResponse | AGENT_CHAT_RESPONSE_TYPE:
        if streaming:
            return await self.chat_engine.astream_chat(query)
        else:
            return await self.chat_engine.achat(query)


def rag_initialize(
    logger: logging.Logger,
    config_path: Path,
    input_from: str | None,
    num_workers: int | None = None,
    recursive: bool = False,
    verbose: bool = False,
) -> tuple[RAGWorkflow, BaseRetriever | None]:
    if verbose:
        logger.info("Load RAG config")
    config = RAGWorkflow.load_config(config_path)

    if verbose:
        logger.info("Read input files...")
    if input_from is not None:
        input_files = read_files(input_from, recursive)
        if verbose:
            count = str(len(input_files)) if input_files else "no"
            logger.info(f"Read {count} input file(s)...done")
    else:
        input_files = None

    if verbose:
        logger.info("Construct RAGWorkflow object")
    rag = RAGWorkflow(config)

    if config.retrieval.embedding is not None:
        # If the input_files is None, the retriever might still load indices
        # from cache or a database
        if verbose:
            logger.info("Load retriever")
        retriever = rag.load_retriever(
            num_workers=num_workers,
            input_files=input_files,
            verbose=verbose,
        )
    else:
        if verbose:
            logger.info("No retriever used")
        retriever = None

    return (rag, retriever)


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

# ret = agent.run(input="Hello!")
