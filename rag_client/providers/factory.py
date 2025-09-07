"""Provider factory implementation for embeddings and LLMs."""

import logging
import subprocess
import uuid
from dataclasses import asdict
from typing import Any, Dict, Optional, Union

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.litellm import LiteLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.perplexity import Perplexity

from ..config.models import (
    EmbeddingConfig,
    HuggingFaceEmbeddingConfig,
    LiteLLMConfig,
    LiteLLMEmbeddingConfig,
    LlamaCPPConfig,
    LLMConfig,
    LMStudioConfig,
    OllamaConfig,
    OllamaEmbeddingConfig,
    OpenAIConfig,
    OpenAIEmbeddingConfig,
    OpenAILikeConfig,
    OpenAILikeEmbeddingConfig,
    OpenRouterConfig,
    PerplexityConfig,
)
from ..utils.helpers import error
from .base import ProviderRegistry

logger = logging.getLogger(__name__)

# Create registries for embeddings and LLMs
embedding_registry = ProviderRegistry[BaseEmbedding]("Embedding")
llm_registry = ProviderRegistry[LLM]("LLM")


# Embedding Provider Implementations

@embedding_registry.decorator("huggingface", {
    "description": "HuggingFace embeddings using transformers",
    "requires_gpu": False,
    "supports_batch": True,
})
class HuggingFaceEmbeddingProvider:
    """HuggingFace embedding provider."""
    
    @staticmethod
    def create(config: HuggingFaceEmbeddingConfig, verbose: bool = False) -> BaseEmbedding:
        """Create HuggingFace embedding instance."""
        return HuggingFaceEmbedding(
            **asdict(config),
            show_progress_bar=verbose,
        )


@embedding_registry.decorator("ollama", {
    "description": "Ollama local embeddings",
    "requires_gpu": False,
    "supports_batch": True,
})
class OllamaEmbeddingProvider:
    """Ollama embedding provider."""
    
    @staticmethod
    def create(config: OllamaEmbeddingConfig, verbose: bool = False) -> BaseEmbedding:
        """Create Ollama embedding instance."""
        return OllamaEmbedding(
            **asdict(config),
            show_progress=verbose,
        )


@embedding_registry.decorator("openai", {
    "description": "OpenAI embeddings API",
    "requires_api_key": True,
    "supports_batch": True,
})
class OpenAIEmbeddingProvider:
    """OpenAI embedding provider."""
    
    @staticmethod
    def create(config: OpenAIEmbeddingConfig, verbose: bool = False) -> BaseEmbedding:
        """Create OpenAI embedding instance."""
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


@embedding_registry.decorator("openai_like", {
    "description": "OpenAI-compatible embeddings API",
    "requires_api_key": True,
    "supports_batch": True,
})
class OpenAILikeEmbeddingProvider:
    """OpenAI-like embedding provider."""
    
    @staticmethod
    def create(config: OpenAILikeEmbeddingConfig, verbose: bool = False) -> BaseEmbedding:
        """Create OpenAI-like embedding instance."""
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


@embedding_registry.decorator("litellm", {
    "description": "LiteLLM unified embeddings interface",
    "requires_api_key": True,
    "supports_batch": True,
})
class LiteLLMEmbeddingProvider:
    """LiteLLM embedding provider."""
    
    @staticmethod
    def create(config: LiteLLMEmbeddingConfig, verbose: bool = False) -> BaseEmbedding:
        """Create LiteLLM embedding instance."""
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


# LLM Provider Implementations

@llm_registry.decorator("ollama", {
    "description": "Ollama local LLM",
    "requires_gpu": False,
    "supports_streaming": True,
})
class OllamaLLMProvider:
    """Ollama LLM provider."""
    
    @staticmethod
    def create(config: OllamaConfig, verbose: bool = False) -> LLM:
        """Create Ollama LLM instance."""
        return Ollama(**asdict(config), show_progress=verbose)


@llm_registry.decorator("openai", {
    "description": "OpenAI API",
    "requires_api_key": True,
    "supports_streaming": True,
    "supports_functions": True,
})
class OpenAILLMProvider:
    """OpenAI LLM provider."""
    
    @staticmethod
    def create(config: OpenAIConfig, verbose: bool = False) -> LLM:
        """Create OpenAI LLM instance."""
        if config.api_key_command is not None:
            config.api_key = subprocess.run(
                config.api_key_command,
                shell=True,
                text=True,
                capture_output=True,
            ).stdout.rstrip("\n")
        return OpenAI(**asdict(config), show_progress=verbose)


@llm_registry.decorator("openai_like", {
    "description": "OpenAI-compatible API",
    "requires_api_key": False,
    "supports_streaming": True,
})
class OpenAILikeLLMProvider:
    """OpenAI-like LLM provider."""
    
    @staticmethod
    def create(config: OpenAILikeConfig, verbose: bool = False) -> LLM:
        """Create OpenAI-like LLM instance."""
        # Handle api_key_command if specified
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
        return OpenAILike(**asdict(config))


@llm_registry.decorator("litellm", {
    "description": "LiteLLM unified LLM interface",
    "requires_api_key": True,
    "supports_streaming": True,
})
class LiteLLMLLMProvider:
    """LiteLLM LLM provider."""
    
    @staticmethod
    def create(config: LiteLLMConfig, verbose: bool = False) -> LLM:
        """Create LiteLLM LLM instance."""
        if config.api_key_command is not None:
            config.api_key = subprocess.run(
                config.api_key_command,
                shell=True,
                text=True,
                capture_output=True,
            ).stdout.rstrip("\n")
        return LiteLLM(**asdict(config))


@llm_registry.decorator("llama_cpp", {
    "description": "LlamaCPP local model",
    "requires_gpu": False,
    "supports_streaming": True,
})
class LlamaCPPLLMProvider:
    """LlamaCPP LLM provider."""
    
    @staticmethod
    def create(config: LlamaCPPConfig, verbose: bool = False) -> LLM:
        """Create LlamaCPP LLM instance."""
        return LlamaCPP(**asdict(config))


@llm_registry.decorator("perplexity", {
    "description": "Perplexity AI with web search",
    "requires_api_key": True,
    "supports_streaming": True,
    "supports_web_search": True,
})
class PerplexityLLMProvider:
    """Perplexity LLM provider."""
    
    @staticmethod
    def create(config: PerplexityConfig, verbose: bool = False) -> LLM:
        """Create Perplexity LLM instance."""
        if config.api_key_command is not None:
            config.api_key = subprocess.run(
                config.api_key_command,
                shell=True,
                text=True,
                capture_output=True,
            ).stdout.rstrip("\n")
        return Perplexity(**asdict(config), show_progress=verbose)


@llm_registry.decorator("openrouter", {
    "description": "OpenRouter multi-model gateway",
    "requires_api_key": True,
    "supports_streaming": True,
})
class OpenRouterLLMProvider:
    """OpenRouter LLM provider."""
    
    @staticmethod
    def create(config: OpenRouterConfig, verbose: bool = False) -> LLM:
        """Create OpenRouter LLM instance."""
        if config.api_key_command is not None:
            config.api_key = subprocess.run(
                config.api_key_command,
                shell=True,
                text=True,
                capture_output=True,
            ).stdout.rstrip("\n")
        return OpenRouter(**asdict(config), show_progress=verbose)


@llm_registry.decorator("lmstudio", {
    "description": "LMStudio local server",
    "requires_api_key": False,
    "supports_streaming": True,
})
class LMStudioLLMProvider:
    """LMStudio LLM provider."""
    
    @staticmethod
    def create(config: LMStudioConfig, verbose: bool = False) -> LLM:
        """Create LMStudio LLM instance."""
        return LMStudio(**asdict(config))


# Factory functions

def create_embedding_provider(
    config: EmbeddingConfig,
    verbose: bool = False
) -> BaseEmbedding:
    """Create embedding provider from configuration.
    
    Args:
        config: Embedding configuration
        verbose: Whether to show verbose output
        
    Returns:
        Initialized embedding provider
        
    Raises:
        ProviderNotFoundError: If provider type is not recognized
        InvalidProviderConfigError: If configuration is invalid
    """
    # Determine provider type from config class
    provider_map = {
        HuggingFaceEmbeddingConfig: "huggingface",
        OllamaEmbeddingConfig: "ollama", 
        OpenAIEmbeddingConfig: "openai",
        OpenAILikeEmbeddingConfig: "openai_like",
        LiteLLMEmbeddingConfig: "litellm",
    }
    
    config_type = type(config)
    provider_name = provider_map.get(config_type)
    
    if not provider_name:
        available = list(provider_map.values())
        error(f"Unknown embedding config type: {config_type.__name__}. Available: {available}")
    
    # Get the provider class
    provider_class = embedding_registry._providers.get(provider_name)
    if not provider_class:
        error(f"Embedding provider not registered: {provider_name}")
    
    # Create the provider
    logger.info(f"Creating embedding provider: {provider_name}")
    return provider_class.create(config, verbose)


def create_llm_provider(
    config: LLMConfig,
    verbose: bool = False
) -> LLM:
    """Create LLM provider from configuration.
    
    Args:
        config: LLM configuration
        verbose: Whether to show verbose output
        
    Returns:
        Initialized LLM provider
        
    Raises:
        ProviderNotFoundError: If provider type is not recognized
        InvalidProviderConfigError: If configuration is invalid
    """
    # Determine provider type from config class
    provider_map = {
        OllamaConfig: "ollama",
        OpenAIConfig: "openai",
        OpenAILikeConfig: "openai_like",
        LiteLLMConfig: "litellm",
        LlamaCPPConfig: "llama_cpp",
        PerplexityConfig: "perplexity",
        OpenRouterConfig: "openrouter",
        LMStudioConfig: "lmstudio",
    }
    
    config_type = type(config)
    provider_name = provider_map.get(config_type)
    
    if not provider_name:
        available = list(provider_map.values())
        error(f"Unknown LLM config type: {config_type.__name__}. Available: {available}")
    
    # Get the provider class
    provider_class = llm_registry._providers.get(provider_name)
    if not provider_class:
        error(f"LLM provider not registered: {provider_name}")
    
    # Create the provider
    logger.info(f"Creating LLM provider: {provider_name}")
    return provider_class.create(config, verbose)


def get_available_embedding_providers() -> Dict[str, Dict[str, Any]]:
    """Get list of available embedding providers with metadata.
    
    Returns:
        Dictionary mapping provider names to their metadata
    """
    providers = {}
    for name in embedding_registry.list_providers():
        info = embedding_registry.get_provider_info(name)
        providers[name] = info
    return providers


def get_available_llm_providers() -> Dict[str, Dict[str, Any]]:
    """Get list of available LLM providers with metadata.
    
    Returns:
        Dictionary mapping provider names to their metadata
    """
    providers = {}
    for name in llm_registry.list_providers():
        info = llm_registry.get_provider_info(name)
        providers[name] = info
    return providers
