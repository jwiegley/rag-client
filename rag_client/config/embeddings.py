"""Pydantic configuration models for embedding providers.

This module contains all embedding provider configurations migrated from
dataclass-wizard to Pydantic with proper validation.
"""

from pathlib import Path
from typing import Any, Literal

from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.embeddings.openai import OpenAIEmbeddingMode, OpenAIEmbeddingModelType
from pydantic import Field, HttpUrl, SecretStr, field_validator

from rag_client.config.base import APIConfig, EmbeddingBaseConfig

# Hardcoded to avoid importing huggingface.base which triggers
# sentence_transformers -> torch -> CUDA at module scope
DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-small-en"


class HuggingFaceEmbeddingConfig(EmbeddingBaseConfig):
    """Configuration for HuggingFace embedding models."""

    model_name: str = Field(
        default=DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
        description="HuggingFace model name or path",
    )
    max_length: int | None = Field(
        default=None, gt=0, le=8192, description="Maximum sequence length"
    )
    query_instruction: str | None = Field(
        default=None, description="Instruction to prepend to queries"
    )
    text_instruction: str | None = Field(
        default=None, description="Instruction to prepend to text passages"
    )
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        gt=0,
        le=1000,
        description="Batch size for embedding generation",
    )
    cache_folder: str | None = Field(
        default=None, description="Path to cache folder for models"
    )
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code from HuggingFace"
    )
    device: (
        Literal["cuda", "cpu", "mps", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] | None
    ) = Field(default=None, description="Device to run model on")
    parallel_process: bool = Field(
        default=False, description="Whether to use parallel processing"
    )
    target_devices: list[str] | None = Field(
        default=None, description="List of target devices for parallel processing"
    )
    tokenizer_name: str | None = Field(
        default=None, description="Optional tokenizer name if different from model"
    )

    @field_validator("cache_folder")
    @classmethod
    def validate_cache_folder(cls, v: str | None) -> str | None:
        """Validate and expand cache folder path."""
        if v:
            path = Path(v).expanduser().resolve()
            return str(path)
        return v

    @field_validator("target_devices")
    @classmethod
    def validate_target_devices(cls, v: list[str] | None, info) -> list[str] | None:
        """Validate target devices when parallel processing is enabled."""
        if info.data.get("parallel_process") and not v:
            raise ValueError(
                "target_devices must be specified when parallel_process is True"
            )
        if v:
            valid_devices = ["cuda", "cpu", "mps"] + [f"cuda:{i}" for i in range(8)]
            for device in v:
                if not any(device.startswith(valid) for valid in valid_devices):
                    raise ValueError(f"Invalid device: {device}")
        return v


class OpenAIEmbeddingConfig(APIConfig, EmbeddingBaseConfig):
    """Configuration for OpenAI embedding models."""

    mode: str = Field(
        default=OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
        description="Embedding mode (similarity or text_search)",
    )
    model: str = Field(
        default=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
        description="OpenAI embedding model name",
    )
    embed_batch_size: int = Field(
        default=100, gt=0, le=2048, description="Batch size for embedding requests"
    )
    dimensions: int | None = Field(
        default=None,
        gt=0,
        le=3072,
        description="Output dimensions (for models that support it)",
    )
    additional_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional parameters to pass to the API"
    )
    reuse_client: bool = Field(
        default=True, description="Whether to reuse the OpenAI client"
    )
    default_headers: dict[str, str] | None = Field(
        default=None, description="Default headers for API requests"
    )
    num_workers: int | None = Field(
        default=None, gt=0, le=100, description="Number of concurrent workers"
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate OpenAI model name."""
        valid_models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-search-ada-query-001",
            "text-search-ada-doc-001",
        ]
        if not any(model in v for model in valid_models):
            # Allow custom models but warn
            pass
        return v

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: int | None, info) -> int | None:
        """Validate dimensions based on model."""
        if v is not None:
            model = info.data.get("model", "")
            if "text-embedding-3" not in model:
                raise ValueError(
                    "dimensions parameter is only supported for text-embedding-3 models"
                )
        return v


class OllamaEmbeddingConfig(EmbeddingBaseConfig):
    """Configuration for Ollama embedding models."""

    model_name: str = Field(..., description="Ollama model name")
    base_url: HttpUrl = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        gt=0,
        le=100,
        description="Batch size for embedding generation",
    )
    request_timeout: float = Field(
        default=60.0, gt=0, le=600, description="Request timeout in seconds"
    )
    ollama_additional_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional Ollama-specific parameters"
    )
    client_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional client parameters"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate Ollama model name."""
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()


class LiteLLMEmbeddingConfig(APIConfig, EmbeddingBaseConfig):
    """Configuration for LiteLLM embedding models."""

    model_name: str = Field(..., description="LiteLLM model identifier")
    embed_batch_size: int = Field(
        default=10, gt=0, le=100, description="Batch size for embedding generation"
    )
    dimensions: int | None = Field(default=None, gt=0, description="Output dimensions")
    additional_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional LiteLLM parameters"
    )
    reuse_client: bool = Field(
        default=True, description="Whether to reuse the LiteLLM client"
    )
    default_headers: dict[str, str] | None = Field(
        default=None, description="Default headers for API requests"
    )
    num_workers: int | None = Field(
        default=None, gt=0, le=50, description="Number of concurrent workers"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate LiteLLM model format."""
        # LiteLLM uses provider/model format
        if "/" not in v and not v.startswith(
            ("openai:", "anthropic:", "cohere:", "replicate:")
        ):
            # Allow but could add warning
            pass
        return v


class OpenAILikeEmbeddingConfig(APIConfig, EmbeddingBaseConfig):
    """Configuration for OpenAI-compatible embedding endpoints."""

    model_name: str = Field(..., description="Model identifier for the endpoint")
    embed_batch_size: int = Field(
        default=10, gt=0, le=100, description="Batch size for embedding generation"
    )
    dimensions: int | None = Field(default=None, gt=0, description="Output dimensions")
    additional_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional API parameters"
    )
    api_key: SecretStr = Field(
        default=SecretStr("fake"),
        description="API key (use 'fake' for local endpoints)",
    )
    reuse_client: bool = Field(default=True, description="Whether to reuse the client")
    default_headers: dict[str, str] | None = Field(
        default=None, description="Default headers for API requests"
    )
    num_workers: int | None = Field(
        default=None, gt=0, le=50, description="Number of concurrent workers"
    )
    add_litellm_session_id: bool = Field(
        default=False, description="Add LiteLLM session ID to requests"
    )
    no_litellm_logging: bool = Field(
        default=False, description="Disable LiteLLM logging"
    )


class VoyageEmbeddingConfig(APIConfig, EmbeddingBaseConfig):
    """Configuration for Voyage AI embedding models."""

    model_name: str = Field(default="voyage-2", description="Voyage model name")
    embed_batch_size: int = Field(
        default=10, gt=0, le=128, description="Batch size for embedding generation"
    )
    input_type: Literal["query", "document"] | None = Field(
        default=None, description="Input type for optimized embeddings"
    )
    truncation: bool = Field(
        default=True, description="Whether to truncate long inputs"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate Voyage model name."""
        valid_models = [
            "voyage-2",
            "voyage-large-2",
            "voyage-code-2",
            "voyage-lite-02-instruct",
        ]
        if v not in valid_models:
            # Allow custom but could warn
            pass
        return v


class CohereEmbeddingConfig(APIConfig, EmbeddingBaseConfig):
    """Configuration for Cohere embedding models."""

    model_name: str = Field(
        default="embed-english-v3.0", description="Cohere model name"
    )
    embed_batch_size: int = Field(
        default=96, gt=0, le=96, description="Batch size for embedding generation"
    )
    input_type: (
        Literal["search_document", "search_query", "classification", "clustering"]
        | None
    ) = Field(default=None, description="Input type for optimized embeddings")
    truncate: Literal["START", "END", "NONE"] | None = Field(
        default="END", description="Truncation strategy for long inputs"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate Cohere model name."""
        valid_models = [
            "embed-english-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-v3.0",
            "embed-multilingual-light-v3.0",
            "embed-english-v2.0",
            "embed-english-light-v2.0",
            "embed-multilingual-v2.0",
        ]
        if v not in valid_models:
            # Allow custom but could warn
            pass
        return v


# LlamaCPP embedding config requires special handling due to llama_cpp imports
# We'll define it but may need to handle the llama_cpp constants carefully
try:
    import llama_cpp

    class LlamaCPPEmbeddingConfig(EmbeddingBaseConfig):
        """Configuration for llama.cpp embedding models."""

        model_path: Path = Field(..., description="Path to the GGUF model file")
        n_gpu_layers: int = Field(
            default=0, ge=0, description="Number of layers to offload to GPU"
        )
        split_mode: int = Field(
            default=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
            description="Split mode for multi-GPU",
        )
        main_gpu: int = Field(default=0, ge=0, description="Main GPU index")
        tensor_split: list[float] | None = Field(
            default=None, description="Tensor split for multi-GPU"
        )
        vocab_only: bool = Field(default=False, description="Load only vocabulary")
        use_mmap: bool = Field(default=True, description="Use memory mapping for model")
        use_mlock: bool = Field(default=False, description="Lock model in memory")
        seed: int = Field(
            default=llama_cpp.LLAMA_DEFAULT_SEED, description="Random seed"
        )
        n_ctx: int = Field(default=512, gt=0, description="Context size")
        n_batch: int = Field(default=512, gt=0, description="Batch size")
        n_threads: int | None = Field(
            default=None, gt=0, description="Number of threads"
        )

        @field_validator("model_path")
        @classmethod
        def validate_model_path(cls, v: Path) -> Path:
            """Validate model path exists and is a file."""
            path = v.expanduser().resolve()
            if not path.exists():
                raise ValueError(f"Model file not found: {path}")
            if not path.is_file():
                raise ValueError(f"Model path is not a file: {path}")
            return path

except ImportError:
    # Define a placeholder if llama_cpp is not installed
    class LlamaCPPEmbeddingConfig(EmbeddingBaseConfig):
        """Configuration for llama.cpp embedding models (llama_cpp not installed)."""

        model_path: Path = Field(..., description="Path to the GGUF model file")
        # Other fields would go here but we can't access the constants


# Export all configuration classes
__all__ = [
    "CohereEmbeddingConfig",
    "HuggingFaceEmbeddingConfig",
    "LiteLLMEmbeddingConfig",
    "LlamaCPPEmbeddingConfig",
    "OllamaEmbeddingConfig",
    "OpenAIEmbeddingConfig",
    "OpenAILikeEmbeddingConfig",
    "VoyageEmbeddingConfig",
]
