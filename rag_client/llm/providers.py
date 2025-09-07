"""LLM provider implementations for RAG client.

This module contains functions for loading various LLM models
from different providers (OpenAI, Ollama, Perplexity, etc.).
"""

import subprocess
import sys
import uuid
from dataclasses import asdict
from typing import NoReturn

from llama_index.core.llms.llm import LLM
from llama_index.llms.litellm import LiteLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.perplexity import Perplexity

from ..config.models import (
    LiteLLMConfig,
    LlamaCPPConfig,
    LLMConfig,
    LMStudioConfig,
    OllamaConfig,
    OpenAIConfig,
    OpenAILikeConfig,
    OpenRouterConfig,
    PerplexityConfig,
)


def error(msg: str) -> NoReturn:
    """Print error message and exit.
    
    Args:
        msg: Error message to display
    """
    print(msg, file=sys.stderr)
    sys.exit(1)


def get_llm(
    config: LLMConfig,
    verbose: bool = False,
) -> LLM:
    """Load an LLM based on configuration.
    
    This function serves as a factory for creating LLM instances based on the
    provided configuration. It supports multiple providers including OpenAI,
    Ollama, Perplexity, LMStudio, OpenRouter, and more.
    
    The function handles:
    - API key resolution from commands if specified
    - Provider-specific configuration options
    - Session ID and logging configuration for certain providers
    
    Args:
        config: LLM configuration object specifying provider and settings
        verbose: Whether to show progress/debug information
        
    Returns:
        LLM: Configured language model ready for use
        
    Raises:
        SystemExit: If LLM cannot be loaded due to invalid configuration
        
    Example:
        >>> from rag_client.config.models import OllamaConfig
        >>> config = OllamaConfig(model="llama2", base_url="http://localhost:11434")
        >>> llm = get_llm(config, verbose=True)
        >>> response = llm.complete("Hello, world!")
    """
    match config:
        case OllamaConfig():
            return Ollama(**asdict(config), show_progress=verbose)
            
        case OpenAILikeConfig():
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
            
        case LiteLLMConfig():
            if config.api_key_command is not None:
                config.api_key = subprocess.run(
                    config.api_key_command,
                    shell=True,
                    text=True,
                    capture_output=True,
                ).stdout.rstrip("\n")
            return LiteLLM(**asdict(config))
            
        case OpenAIConfig():
            if config.api_key_command is not None:
                config.api_key = subprocess.run(
                    config.api_key_command,
                    shell=True,
                    text=True,
                    capture_output=True,
                ).stdout.rstrip("\n")
            return OpenAI(**asdict(config), show_progress=verbose)
            
        case LlamaCPPConfig():
            return LlamaCPP(**asdict(config))
            
        case PerplexityConfig():
            if config.api_key_command is not None:
                config.api_key = subprocess.run(
                    config.api_key_command,
                    shell=True,
                    text=True,
                    capture_output=True,
                ).stdout.rstrip("\n")
            return Perplexity(**asdict(config), show_progress=verbose)
            
        case OpenRouterConfig():
            if config.api_key_command is not None:
                config.api_key = subprocess.run(
                    config.api_key_command,
                    shell=True,
                    text=True,
                    capture_output=True,
                ).stdout.rstrip("\n")
            return OpenRouter(**asdict(config), show_progress=verbose)
            
        case LMStudioConfig():
            return LMStudio(**asdict(config))
            
        case _:
            error(f"Unknown LLM configuration type: {config}")


def realize_llm(
    config: LLMConfig | None,
    verbose: bool = False,
) -> LLM | NoReturn:
    """Load an LLM or error if not possible.
    
    Args:
        config: Optional LLM configuration
        verbose: Whether to show progress
        
    Returns:
        LLM: Configured language model
        
    Raises:
        SystemExit: If LLM cannot be loaded or config is None
    """
    if config is None:
        error("No LLM configuration provided")
    
    try:
        return get_llm(config=config, verbose=verbose)
    except Exception as e:
        error(f"Failed to start LLM: {config}\nError: {e}")
