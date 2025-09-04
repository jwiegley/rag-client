"""Embedding provider implementations for RAG client.

This module contains functions and classes for loading various embedding models
from different providers (HuggingFace, OpenAI, Ollama, etc.).
"""

import subprocess
import sys
import uuid
from dataclasses import asdict
from typing import Any, Literal, NoReturn, override

import llama_cpp
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from ..config.models import (
    EmbeddingConfig,
    HuggingFaceEmbeddingConfig,
    LiteLLMEmbeddingConfig,
    LlamaCPPEmbeddingConfig,
    OllamaEmbeddingConfig,
    OpenAIEmbeddingConfig,
    OpenAILikeEmbeddingConfig,
)


def error(msg: str) -> NoReturn:
    """Print error message and exit.
    
    Args:
        msg: Error message to display
    """
    print(msg, file=sys.stderr)
    sys.exit(1)


def flatten_floats(data: list[float] | list[list[float]]) -> list[float]:
    """Flatten a nested list of floats.
    
    Args:
        data: Either a flat list of floats or a nested list
        
    Returns:
        Flat list of floats
    """
    if not data:
        return []
    # Check if the first element is a float (covers the list[float] case)
    if isinstance(data[0], float):
        return data  # type: ignore
    # Otherwise, it's a list of lists
    return [item for sublist in data for item in sublist]  # type: ignore


class LlamaCPPEmbedding(BaseEmbedding):
    """Custom embedding implementation for LlamaCPP models.
    
    This class provides embedding functionality using llama.cpp models loaded
    locally. It supports GPU acceleration and various configuration options
    for optimal performance.
    
    Attributes:
        _model: The underlying llama_cpp.Llama model instance
        
    Example:
        >>> config = LlamaCPPEmbeddingConfig(
        ...     model_path="/path/to/model.gguf",
        ...     n_ctx=512
        ... )
        >>> embedding = LlamaCPPEmbedding(**asdict(config))
        >>> vec = embedding.get_text_embedding("Hello world")
    """
    
    def __init__(self, model_path: str, **kwargs):
        """Initialize LlamaCPP embedding model.
        
        Args:
            model_path: Path to the GGUF model file
            **kwargs: Additional configuration options including:
                - verbose: Whether to show progress (default: False)
                - n_ctx: Context window size (default: 512)
                - n_batch: Batch size for processing (default: 512)
                - n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        """
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
            verbose=kwargs.get("verbose", False),
        )
    
    @override
    def _get_text_embedding(self, text: str) -> Embedding:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        response = self._model.create_embedding(text)
        return flatten_floats(response["data"][0]["embedding"])
    
    @override
    def _get_query_embedding(self, query: str) -> Embedding:
        """Generate embedding for a query text.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        response = self._model.create_embedding(query)
        return flatten_floats(response["data"][0]["embedding"])
    
    @override
    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        response = self._model.create_embedding(texts)
        return [flatten_floats(item["embedding"]) for item in response["data"]]
    
    @override
    async def _aget_text_embedding(self, text: str):
        """Async version of text embedding (delegates to sync).
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        return self._get_text_embedding(text)
    
    @override
    async def _aget_query_embedding(self, query: str):
        """Async version of query embedding (delegates to sync).
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        return self._get_query_embedding(query)


def get_embedding_model(
    config: EmbeddingConfig,
    verbose: bool = False,
) -> BaseEmbedding:
    """Load an embedding model based on configuration.
    
    Args:
        config: Embedding configuration
        verbose: Whether to show progress
        
    Returns:
        BaseEmbedding: Configured embedding model
        
    Raises:
        SystemExit: If embedding model cannot be loaded
    """
    match config:
        case HuggingFaceEmbeddingConfig():
            return HuggingFaceEmbedding(
                **asdict(config),
                show_progress_bar=verbose,
            )
        case OllamaEmbeddingConfig():
            return OllamaEmbedding(
                **asdict(config),
                show_progress=verbose,
            )
        case LlamaCPPEmbeddingConfig():
            return LlamaCPPEmbedding(
                **asdict(config),
                show_progress=verbose,
            )
        case OpenAIEmbeddingConfig():
            if config.api_key_command is not None:
                config.api_key = subprocess.run(
                    config.api_key_command,
                    shell=True,
                    text=True,
                    capture_output=True,
                ).stdout.rstrip("\n")
            return OpenAIEmbedding(
                **asdict(config),
                show_progress=verbose,
            )
        case OpenAILikeEmbeddingConfig():
            if config.api_key_command is not None:
                config.api_key = subprocess.run(
                    config.api_key_command,
                    shell=True,
                    text=True,
                    capture_output=True,
                ).stdout.rstrip("\n")
            
            extra_body = {}
            if config.add_litellm_session_id:
                extra_body["litellm_session_id"] = str(uuid.uuid1())
            if config.no_litellm_logging:
                extra_body["no-log"] = True
            
            if config.additional_kwargs is None:
                config.additional_kwargs = {"extra_body": extra_body}
            elif "extra_body" not in config.additional_kwargs:
                config.additional_kwargs["extra_body"] = extra_body
            else:
                config.additional_kwargs["extra_body"] = (
                    config.additional_kwargs["extra_body"] | extra_body
                )
            
            return OpenAILikeEmbedding(
                show_progress=verbose,
                **asdict(config),
            )
        case LiteLLMEmbeddingConfig():
            if config.api_key_command is not None:
                config.api_key = subprocess.run(
                    config.api_key_command,
                    shell=True,
                    text=True,
                    capture_output=True,
                ).stdout.rstrip("\n")
            return LiteLLMEmbedding(
                **asdict(config),
            )
        case _:
            error(f"Unknown embedding configuration type: {config}")