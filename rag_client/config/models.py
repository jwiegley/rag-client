"""Configuration models for RAG client.

This module contains all YAMLWizard configuration dataclasses used throughout
the RAG client application.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union, final

import llama_cpp
import llama_index.llms.lmstudio.base
import llama_index.llms.mlx.base
import llama_index.llms.ollama.base
import llama_index.llms.openrouter.base
from dataclass_wizard import JSONWizard, YAMLWizard


# Configure dataclass_wizard to use 'type' field for union discrimination
@final
class GlobalJSONMeta(JSONWizard.Meta):
    tag_key = "type"
    auto_assign_tags = True
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_SIMILARITY_TOP_K,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.embeddings.huggingface.base import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
)
from llama_index.embeddings.openai import (
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
)
from llama_index.llms.openai.base import DEFAULT_OPENAI_MODEL

# Logging Configuration

@dataclass
class LoggingConfig(YAMLWizard):
    """Configuration for application logging.
    
    Controls logging behavior throughout the RAG client application,
    including log levels, output destinations, and rotation policies.
    
    Attributes:
        level: Logging level. Controls verbosity of output.
            Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.
            Default: INFO (general operational messages).
        log_file: Optional path to log file. If None, logs to console only.
            Example: "/var/log/rag-client.log"
        format: Custom log format string. If None, uses default format.
            Example: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        rotate_logs: If True, enables log rotation to prevent large files.
        max_bytes: Maximum log file size before rotation (default: 10MB).
            Only used when rotate_logs=True.
        backup_count: Number of rotated log files to keep (default: 5).
            Only used when rotate_logs=True.
        console_output: If True, also outputs logs to console/stderr.
    
    Example YAML:
        ```yaml
        logging:
          level: DEBUG
          log_file: logs/rag.log
          rotate_logs: true
          max_bytes: 5242880  # 5MB
          backup_count: 10
          console_output: true
        ```
    """
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: Optional[str] = None
    format: Optional[str] = None
    rotate_logs: bool = False
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    console_output: bool = True


# Embedding Configurations

@dataclass
class HuggingFaceEmbeddingConfig(YAMLWizard):
    """HuggingFace embedding configuration.
    
    Configures HuggingFace Transformers models for generating text embeddings.
    Supports both local and remote models from HuggingFace Hub.
    
    Attributes:
        model_name: HuggingFace model identifier or local path.
            Default: "BAAI/bge-small-en-v1.5" (efficient English model).
            Popular options: "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-large-en-v1.5", "intfloat/e5-large-v2".
        max_length: Maximum sequence length. If None, uses model's default.
            Longer texts are truncated. Typical: 512 tokens.
        query_instruction: Instruction prepended to queries for asymmetric models.
            Example: "Represent this question for searching:"
        text_instruction: Instruction prepended to documents.
            Example: "Represent this document for retrieval:"
        normalize: If True, normalizes embeddings to unit length.
            Recommended for cosine similarity.
        embed_batch_size: Number of texts to embed simultaneously.
            Higher values faster but use more memory. Default: 10.
        cache_folder: Local directory to cache downloaded models.
            If None, uses HuggingFace default (~/.cache/huggingface).
        trust_remote_code: If True, allows executing remote code from model repo.
            Security risk - only enable for trusted models.
        device: Device to run model on ("cuda", "cpu", "mps").
            If None, auto-detects available hardware.
        parallel_process: If True, enables multi-GPU processing.
        target_devices: List of specific GPU devices for parallel processing.
            Example: ["cuda:0", "cuda:1"]
    
    Example YAML:
        ```yaml
        embedding:
          type: huggingface
          model_name: BAAI/bge-base-en-v1.5
          max_length: 512
          query_instruction: "Represent this sentence for searching:"
          normalize: true
          embed_batch_size: 32
          device: cuda
        ```
    
    Note:
        - First run downloads model (can be several GB)
        - Consider model size vs accuracy tradeoff
        - Asymmetric models need different instructions for queries/docs
    """
    model_name: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
    max_length: Optional[int] = None
    query_instruction: Optional[str] = None
    text_instruction: Optional[str] = None
    normalize: bool = True
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    cache_folder: Optional[str] = None
    trust_remote_code: bool = False
    device: Optional[str] = None
    parallel_process: bool = False
    target_devices: Optional[List[str]] = None


@dataclass
class OllamaEmbeddingConfig(YAMLWizard):
    """Ollama embedding configuration.
    
    Configures Ollama for generating embeddings using locally-hosted models.
    Ollama provides easy deployment of open-source models.
    
    Attributes:
        model_name: Name of the Ollama model to use for embeddings.
            Must be pulled first: `ollama pull <model>`
            Examples: "nomic-embed-text", "mxbai-embed-large".
        base_url: Ollama server URL (default: "http://localhost:11434").
            Change if running Ollama on different host/port.
        embed_batch_size: Number of texts to embed per request.
            Default: 10. Adjust based on model and memory.
        ollama_additional_kwargs: Extra parameters for Ollama API.
            Example: {"num_thread": 8, "temperature": 0}
        client_kwargs: HTTP client configuration.
            Example: {"timeout": 30, "verify_ssl": false}
    
    Example YAML:
        ```yaml
        embedding:
          type: ollama
          model_name: nomic-embed-text
          base_url: http://localhost:11434
          embed_batch_size: 20
          ollama_additional_kwargs:
            num_thread: 4
        ```
    
    Prerequisites:
        1. Install Ollama: https://ollama.ai
        2. Pull embedding model: `ollama pull nomic-embed-text`
        3. Ensure Ollama service is running
    
    Note:
        - Ollama models run entirely locally (privacy-preserving)
        - Performance depends on local hardware
        - Some models optimized for specific hardware (e.g., Apple Silicon)
    """
    model_name: str
    base_url: str = "http://localhost:11434"
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    ollama_additional_kwargs: Optional[Dict[str, Any]] = None
    client_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class OpenAIEmbeddingConfig(YAMLWizard):
    """OpenAI embedding configuration."""
    mode: str = OpenAIEmbeddingMode.TEXT_SEARCH_MODE
    model: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
    embed_batch_size: int = 100
    dimensions: Optional[int] = None
    additional_kwargs: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None
    api_key_command: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    max_retries: int = 10
    timeout: float = 60.0
    reuse_client: bool = True
    default_headers: Optional[Dict[str, str]] = None
    num_workers: Optional[int] = None


@dataclass
class OpenAILikeEmbeddingConfig(YAMLWizard):
    """OpenAI-like embedding configuration."""
    model_name: str
    embed_batch_size: int = 10
    dimensions: Optional[int] = None
    additional_kwargs: Optional[Dict[str, Any]] = None
    api_key: str = "fake"
    api_key_command: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    max_retries: int = 10
    timeout: float = 60.0
    reuse_client: bool = True
    default_headers: Optional[Dict[str, str]] = None
    num_workers: Optional[int] = None
    add_litellm_session_id: bool = False
    no_litellm_logging: bool = False


@dataclass
class LiteLLMEmbeddingConfig(YAMLWizard):
    """LiteLLM embedding configuration."""
    model_name: str
    embed_batch_size: int = 10
    dimensions: Optional[int] = None
    additional_kwargs: Optional[Dict[str, Any]] = None
    api_key: str = "fake"
    api_key_command: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    max_retries: int = 10
    timeout: float = 60.0
    reuse_client: bool = True
    default_headers: Optional[Dict[str, str]] = None
    num_workers: Optional[int] = None


@dataclass
class LlamaCPPEmbeddingConfig(YAMLWizard):
    """LlamaCPP embedding configuration."""
    model_path: Path
    n_gpu_layers: int = 0
    split_mode: int = llama_cpp.LLAMA_SPLIT_MODE_LAYER
    main_gpu: int = 0
    tensor_split: Optional[List[float]] = None
    rpc_servers: Optional[str] = None
    vocab_only: bool = False
    use_mmap: bool = True
    use_mlock: bool = False
    kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None
    # Context Params
    seed: int = llama_cpp.LLAMA_DEFAULT_SEED
    n_ctx: int = 512
    n_batch: int = 512
    n_ubatch: int = 512
    n_threads: int | None = None
    n_threads_batch: int | None = None
    rope_scaling_type: int = llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
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
    # Speculative Decoding
    # draft_model: llama_cpp.LlamaDraftModel | None = None
    # Tokenizer Override
    # tokenizer: llama_cpp.BaseLlamaTokenizer | None = None
    # KV cache quantization
    type_k: int | None = None
    type_v: int | None = None
    # Misc
    spm_infill: bool = False


EmbeddingConfig: TypeAlias = (
    HuggingFaceEmbeddingConfig
    | OllamaEmbeddingConfig
    | OpenAIEmbeddingConfig
    | OpenAILikeEmbeddingConfig
    | LiteLLMEmbeddingConfig
    | LlamaCPPEmbeddingConfig
)


def embedding_model(config: EmbeddingConfig) -> str:
    """Get model name from embedding configuration.
    
    Args:
        config: Embedding configuration
        
    Returns:
        Model name string
    """
    match config:
        case HuggingFaceEmbeddingConfig():
            return config.model_name
        case OllamaEmbeddingConfig():
            return config.model_name
        case LlamaCPPEmbeddingConfig():
            return str(config.model_path)
        case OpenAIEmbeddingConfig():
            return config.model
        case OpenAILikeEmbeddingConfig():
            return config.model_name
        case LiteLLMEmbeddingConfig():
            return config.model_name


# LLM Configurations

@dataclass
class OllamaConfig(YAMLWizard):
    """Ollama LLM configuration."""
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.75
    context_window: int = DEFAULT_CONTEXT_WINDOW
    request_timeout: Optional[float] = llama_index.llms.ollama.base.DEFAULT_REQUEST_TIMEOUT
    prompt_key: str = "prompt"
    json_mode: bool = False
    # additional_kwargs: Dict[str, Any] = field(default_factory=dict)
    is_function_calling_model: bool = True
    keep_alive: Optional[Union[float, str]] = None


@dataclass
class OpenAIConfig(YAMLWizard):
    """OpenAI LLM configuration."""
    model: str = DEFAULT_OPENAI_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: Optional[int] = None
    additional_kwargs: Optional[Dict[str, Any]] = None
    max_retries: int = 3
    timeout: float = 60.0
    reuse_client: bool = True
    api_key: str = "fake"
    api_key_command: Optional[str] = None
    api_base: Optional[str] = None
    api_version: str = ""
    default_headers: Optional[Dict[str, str]] = None
    # base class
    system_prompt: Optional[str] = None
    # output_parser: Optional[BaseOutputParser] = None
    strict: bool = False
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    modalities: Optional[List[str]] = None
    audio_config: Optional[Dict[str, Any]] = None


@dataclass
class OpenAILikeConfig(OpenAIConfig):
    context_window: int = DEFAULT_CONTEXT_WINDOW
    is_chat_model: bool = False
    is_function_calling_model: bool = False
    add_litellm_session_id: bool = False
    no_litellm_logging: bool = False


@dataclass
class LiteLLMConfig(OpenAIConfig):
    context_window: int = DEFAULT_CONTEXT_WINDOW
    is_chat_model: bool = False
    is_function_calling_model: bool = False


@dataclass
class LlamaCPPConfig(YAMLWizard):
    """LlamaCPP LLM configuration."""
    model_url: Optional[str] = None
    model_path: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_new_tokens: int = DEFAULT_NUM_OUTPUTS
    context_window: int = DEFAULT_CONTEXT_WINDOW
    generate_kwargs: Optional[Dict[str, Any]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    verbose: bool = True  # Default verbosity for LlamaCPP
    system_prompt: Optional[str] = None
    # output_parser: Optional[BaseOutputParser] = None


@dataclass
class PerplexityConfig(YAMLWizard):
    """Perplexity LLM configuration."""
    model: str = "sonar-pro"
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    api_key_command: Optional[str] = None
    api_base: Optional[str] = "https://api.perplexity.ai"
    additional_kwargs: Optional[Dict[str, Any]] = None
    max_retries: int = 10
    context_window: Optional[int] = None
    system_prompt: Optional[str] = None
    # output_parser: Optional[BaseOutputParser] = None
    enable_search_classifier: bool = False


@dataclass
class OpenRouterConfig(YAMLWizard):
    """OpenRouter LLM configuration."""
    model: str = llama_index.llms.openrouter.base.DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_NUM_OUTPUTS
    additional_kwargs: Optional[Dict[str, Any]] = None
    max_retries: int = 5
    api_base: Optional[str] = llama_index.llms.openrouter.base.DEFAULT_API_BASE
    api_key: Optional[str] = None
    api_key_command: Optional[str] = None


@dataclass
class LMStudioConfig(YAMLWizard):
    """LMStudio LLM configuration."""
    model_name: str
    system_prompt: Optional[str] = None
    # output_parser: Optional[BaseOutputParser] = None
    base_url: str = "http://localhost:1234/v1"
    context_window: int = DEFAULT_CONTEXT_WINDOW
    request_timeout: float = llama_index.llms.lmstudio.base.DEFAULT_REQUEST_TIMEOUT
    num_output: int = DEFAULT_NUM_OUTPUTS
    is_chat_model: bool = True
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = 120
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLXLLMConfig(YAMLWizard):
    """MLX LLM configuration."""
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_new_tokens: int = DEFAULT_NUM_OUTPUTS
    # query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}"
    model_name: str = llama_index.llms.mlx.base.DEFAULT_MLX_MODEL
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    tokenizer_outputs_to_remove: Optional[List[str]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    generate_kwargs: Optional[Dict[str, Any]] = None
    system_prompt: str = ""
    # output_parser: Optional[BaseOutputParser] = None


LLMConfig: TypeAlias = (
    OllamaConfig
    | OpenAILikeConfig
    | LiteLLMConfig
    | OpenAIConfig
    | LlamaCPPConfig
    | PerplexityConfig
    | OpenRouterConfig
    | LMStudioConfig
    | MLXLLMConfig
)


def llm_model(config: LLMConfig) -> str:
    """Get model name from LLM configuration.
    
    Args:
        config: LLM configuration
        
    Returns:
        Model name string
    """
    match config:
        case OllamaConfig():
            return config.model
        case OpenAILikeConfig():
            return config.model
        case LiteLLMConfig():
            return config.model
        case OpenAIConfig():
            return config.model
        case LlamaCPPConfig():
            msg = "<unknown LlamaCPP model>"
            return config.model_path or config.model_url or msg
        case PerplexityConfig():
            return config.model
        case OpenRouterConfig():
            return config.model
        case LMStudioConfig():
            return config.model_name
        case MLXLLMConfig():
            return config.model_name


# Index and Processing Configurations

@dataclass
class KeywordsConfig(YAMLWizard):
    """Keywords collection configuration."""
    collect: bool = False
    llm: Optional[LLMConfig] = None


@dataclass
class SentenceSplitterConfig(YAMLWizard):
    """Sentence splitter configuration."""
    chunk_size: int = 1024
    chunk_overlap: int = 200
    include_metadata: bool = True


@dataclass
class SentenceWindowSplitterConfig(YAMLWizard):
    """Sentence window splitter configuration."""
    window_size: int = 3
    window_metadata_key: str = "window"
    original_text_metadata_key: str = "original_text"


@dataclass
class SemanticSplitterConfig(YAMLWizard):
    """Semantic splitter configuration."""
    embedding: EmbeddingConfig
    buffer_size: int = 1
    breakpoint_percentile_threshold: int = 95
    include_metadata: bool = True


@dataclass
class JSONNodeParserConfig(YAMLWizard):
    """JSON node parser configuration."""
    include_metadata: bool = True
    include_prev_next_rel: bool = True


@dataclass
class CodeSplitterConfig(YAMLWizard):
    """Code splitter configuration."""
    language: str
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


# Extractor Configurations

@dataclass
class KeywordExtractorConfig(YAMLWizard):
    """Keyword extractor configuration."""
    llm: LLMConfig
    keywords: int = 5


@dataclass
class SummaryExtractorConfig(YAMLWizard):
    """Summary extractor configuration."""
    llm: LLMConfig
    # ["self"]
    # summaries: List[str] = field(default_factory=list)
    summaries: Optional[List[str]] = None


@dataclass
class TitleExtractorConfig(YAMLWizard):
    """Title extractor configuration."""
    llm: LLMConfig
    nodes: int = 5


@dataclass
class QuestionsAnsweredExtractorConfig(YAMLWizard):
    """Questions answered extractor configuration."""
    llm: LLMConfig
    questions: int = 1


ExtractorConfig: TypeAlias = (
    KeywordExtractorConfig
    | SummaryExtractorConfig
    | TitleExtractorConfig
    | QuestionsAnsweredExtractorConfig
)


# Evaluator Configurations

@dataclass
class RelevancyEvaluatorConfig(YAMLWizard):
    """Relevancy evaluator configuration."""
    llm: LLMConfig


@dataclass
class GuidelineConfig(YAMLWizard):
    """Guideline evaluator configuration."""
    llm: LLMConfig
    guidelines: str = DEFAULT_GUIDELINES


EvaluatorConfig: TypeAlias = RelevancyEvaluatorConfig | GuidelineConfig


# Query Engine Configurations

@dataclass
class CitationQueryEngineConfig(YAMLWizard):
    """Citation query engine configuration."""
    chunk_size: int = 512
    chunk_overlap: int = 20


@dataclass
class RetrieverQueryEngineConfig(YAMLWizard):
    """Retriever query engine configuration."""
    response_mode: ResponseMode = ResponseMode.REFINE


@dataclass
class SimpleQueryEngineConfig(YAMLWizard):
    """Simple query engine configuration."""
    pass


BaseQueryEngineConfig: TypeAlias = (
    SimpleQueryEngineConfig | CitationQueryEngineConfig | RetrieverQueryEngineConfig
)


@dataclass
class MultiStepQueryEngineConfig(YAMLWizard):
    """Multi-step query engine configuration."""
    engine: BaseQueryEngineConfig


@dataclass
class RetrySourceQueryEngineConfig(YAMLWizard):
    """Retry source query engine configuration."""
    llm: LLMConfig
    evaluator: EvaluatorConfig
    engine: BaseQueryEngineConfig


@dataclass
class RetryQueryEngineConfig(YAMLWizard):
    """Retry query engine configuration."""
    evaluator: EvaluatorConfig
    engine: BaseQueryEngineConfig


QueryEngineConfig: TypeAlias = (
    BaseQueryEngineConfig
    | MultiStepQueryEngineConfig
    | RetrySourceQueryEngineConfig
    | RetryQueryEngineConfig
)


# Chat Engine Configurations

@dataclass
class SimpleChatEngineConfig(YAMLWizard):
    """Simple chat engine configuration."""
    pass


@dataclass
class SimpleContextChatEngineConfig(YAMLWizard):
    """Simple context chat engine configuration."""
    context_window: int = DEFAULT_CONTEXT_WINDOW


@dataclass
class ContextChatEngineConfig(YAMLWizard):
    """Context chat engine configuration."""
    pass


@dataclass
class CondensePlusContextChatEngineConfig(YAMLWizard):
    """Condense plus context chat engine configuration."""
    skip_condense: bool = False


ChatEngineConfig: TypeAlias = (
    SimpleChatEngineConfig
    | SimpleContextChatEngineConfig
    | ContextChatEngineConfig
    | CondensePlusContextChatEngineConfig
)


# Vector Store Configurations

@dataclass
class SimpleVectorStoreConfig(YAMLWizard):
    """Simple vector store configuration."""
    pass


@dataclass
class PostgresVectorStoreConfig(YAMLWizard):
    """PostgreSQL vector store configuration."""
    connection: str
    hybrid_search: bool = False
    dimensions: int = 512
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    hnsw_dist_method: str = "vector_cosine_ops"


VectorStoreConfig: TypeAlias = SimpleVectorStoreConfig | PostgresVectorStoreConfig


# Retriever Configurations

@dataclass
class FusionRetrieverConfig(YAMLWizard):
    """Fusion retriever configuration."""
    llm: LLMConfig
    num_queries: int = 1  # set this to 1 to disable query generation
    mode: FUSION_MODES = FUSION_MODES.RELATIVE_SCORE


# Main Configurations

@dataclass
class RetrievalConfig(YAMLWizard):
    """Retrieval configuration.
    
    Defines the complete retrieval pipeline for document indexing and search.
    Coordinates embedding generation, storage, chunking, and retrieval strategies.
    
    Attributes:
        llm: Language model configuration for advanced retrieval features.
            Used for query expansion, re-ranking, and extraction.
        embedding: Embedding model configuration for vectorization.
            Defines how text is converted to searchable vectors.
        vector_store: Storage backend for embeddings and documents.
            Options: SimpleVectorStore (memory), PostgresVectorStore (persistent).
        splitter: Text splitting strategy for chunking documents.
            If None, uses default SentenceSplitter.
        extractors: List of metadata extractors to enrich documents.
            Can extract keywords, summaries, titles, Q&A pairs.
        keywords: Configuration for keyword-based (sparse) retrieval.
            Enables hybrid search when combined with vector search.
        top_k: Number of top results for vector similarity search.
            Default: 2. Typical range: 3-20 depending on use case.
        sparse_top_k: Number of results for keyword search.
            Default: 10. Used in hybrid retrieval scenarios.
        fusion: Configuration for combining multiple retrievers.
            Enables advanced fusion strategies for better results.
    
    Example YAML:
        ```yaml
        retrieval:
          llm:
            type: ollama
            model: llama2
          embedding:
            type: huggingface
            model_name: BAAI/bge-small-en-v1.5
          vector_store:
            type: simple
          splitter:
            type: sentence
            chunk_size: 512
            chunk_overlap: 50
          top_k: 5
          keywords:
            collect: true
        ```
    
    Note:
        - LLM is required even for basic retrieval (used for processing)
        - Embedding model must match vector store dimensions
        - Consider memory usage with large document sets
    """
    embed_individually: bool = False
    embedding: Optional[EmbeddingConfig] = None
    keywords: Optional[KeywordsConfig] = None
    splitter: Optional[SplitterConfig] = None
    extractors: Optional[List[ExtractorConfig]] = None
    vector_store: Optional[VectorStoreConfig] = None
    fusion: Optional[FusionRetrieverConfig] = None


@dataclass
class QueryConfig(YAMLWizard):
    """Query configuration.
    
    Configures the query processing engine for one-shot Q&A operations.
    Defines how queries are processed, contextualized, and answered.
    
    Attributes:
        engine: Query engine configuration defining processing strategy.
            Options include:
            - SimpleQueryEngine: Direct LLM responses without retrieval
            - RetrieverQueryEngine: RAG with retrieval and synthesis
            - CitationQueryEngine: Includes source citations
            - RetryQueryEngine: Automatic retry on failures
            If None, defaults to SimpleQueryEngine.
        llm: Language model for generating responses.
            Inherited from engine configuration.
    
    Example YAML:
        ```yaml
        query:
          engine:
            type: retriever
            response_mode: compact
          llm:
            type: openai
            model: gpt-3.5-turbo
            temperature: 0.7
        ```
    
    Query Engines:
        - Simple: Direct LLM response, no retrieval
        - Retriever: Retrieves context then generates
        - Citation: Adds source references to responses
        - Retry: Handles failures with automatic retries
        - RetrySource: Retries with different sources
    
    Note:
        - Query operations are stateless (no history)
        - For conversational interactions, use ChatConfig
        - Engine choice impacts response quality and latency
    """
    llm: LLMConfig
    engine: Optional[QueryEngineConfig] = None
    retries: bool = False
    source_retries: bool = False
    show_citations: bool = False
    evaluator_llm: Optional[LLMConfig] = None


@dataclass
class ChatConfig(YAMLWizard):
    """Chat configuration.
    
    Configures the conversational AI system with memory and context management.
    Enables multi-turn conversations with optional RAG enhancement.
    
    Attributes:
        engine: Chat engine configuration defining conversation strategy.
            Options include:
            - SimpleChatEngine: Basic chat without retrieval
            - ContextChatEngine: RAG-enhanced with retrieval
            - CondensePlusContextChatEngine: Condenses history + retrieval
            If None, defaults to SimpleChatEngine.
        buffer: Number of recent messages to keep in context.
            Default: 10. Prevents context overflow.
        summary_buffer: Number of messages before summarization.
            If set, older messages are summarized to save tokens.
            If None, no summarization (messages dropped after buffer).
        keep_history: If True, persists chat history to disk.
            History saved to ~/.config/rag-client/chat_store.json.
        default_user: Default username for chat sessions.
            Used for history tracking and personalization.
    
    Example YAML:
        ```yaml
        chat:
          engine:
            type: context
            system_prompt: "You are a helpful assistant."
          buffer: 20
          summary_buffer: 50
          keep_history: true
          default_user: "user"
          llm:
            type: openai
            model: gpt-4
            temperature: 0.8
        ```
    
    Chat Engines:
        - Simple: Direct conversation, no retrieval
        - Context: Retrieves relevant docs for each turn
        - CondensePlusContext: Condenses chat + retrieves
    
    Note:
        - Buffer management crucial for long conversations
        - History persistence enables cross-session continuity
        - Context engines significantly improve factual responses
    """
    llm: LLMConfig
    engine: Optional[ChatEngineConfig] = None
    default_user: str = "user"
    summarize: bool = False
    keep_history: bool = False


@dataclass
class Config(YAMLWizard):
    """Main application configuration.
    
    Root configuration object that orchestrates all RAG client components.
    Loaded from YAML files to configure the entire application behavior.
    
    Attributes:
        retrieval: Configuration for document indexing and retrieval.
            Required for all RAG operations. Defines embedding models,
            storage backends, and retrieval strategies.
        query: Configuration for one-shot Q&A operations.
            Optional. If provided, enables query command functionality.
        chat: Configuration for conversational interactions.
            Optional. If provided, enables interactive chat mode.
        logging: Application logging configuration.
            Optional. Controls log levels, outputs, and rotation.
    
    Complete Example YAML:
        ```yaml
        # Retrieval configuration (required)
        retrieval:
          embedding:
            type: huggingface
            model_name: BAAI/bge-small-en-v1.5
          llm:
            type: ollama
            model: llama2
            base_url: http://localhost:11434
          vector_store:
            type: postgres
            connection: postgresql://user:pass@localhost/ragdb
            dimensions: 384
          splitter:
            type: sentence
            chunk_size: 512
            chunk_overlap: 50
          top_k: 5
        
        # Query configuration (optional)
        query:
          engine:
            type: retriever
            response_mode: tree_summarize
          llm:
            type: openai
            model: gpt-3.5-turbo
        
        # Chat configuration (optional)
        chat:
          engine:
            type: context
          buffer: 20
          keep_history: true
          llm:
            type: openai
            model: gpt-4
        
        # Logging configuration (optional)
        logging:
          level: INFO
          log_file: logs/rag.log
          rotate_logs: true
        ```
    
    Usage:
        ```python
        config = Config.from_yaml_file("config.yaml")
        workflow = RAGWorkflow(logger, config)
        ```
    
    Note:
        - At minimum, retrieval configuration is required
        - Query and chat can use different LLMs for cost optimization
        - YAML structure must match dataclass hierarchy exactly
    """
    retrieval: RetrievalConfig
    query: Optional[QueryConfig] = None
    chat: Optional[ChatConfig] = None
    logging: Optional[LoggingConfig] = None