"""Pydantic configuration models for LLM providers.

This module contains all LLM provider configurations migrated from
dataclass-wizard to Pydantic with proper validation.
"""

from typing import Any, Dict, List, Literal, Optional

# Import constants from llama-index
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.llms.openai.base import DEFAULT_OPENAI_MODEL
from pydantic import Field, HttpUrl, SecretStr, field_validator

from rag_client.config.base import APIConfig, LLMBaseConfig


class OpenAILLMConfig(APIConfig, LLMBaseConfig):
    """Configuration for OpenAI LLM models."""
    
    model_name: str = Field(
        default=DEFAULT_OPENAI_MODEL,
        description="OpenAI model name"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=DEFAULT_NUM_OUTPUTS,
        gt=0,
        le=128000,
        description="Maximum tokens to generate"
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        gt=0,
        description="Context window size"
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream responses"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    response_format: Optional[Dict[str, str]] = Field(
        default=None,
        description="Response format (e.g., {'type': 'json_object'})"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for deterministic generation"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        default=None,
        description="Token logit bias"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend"
    )
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate OpenAI model name."""
        valid_prefixes = ['gpt-3.5', 'gpt-4', 'text-davinci', 'text-curie', 'text-babbage', 'text-ada']
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            # Allow custom models but could warn
            pass
        return v
    
    @field_validator('response_format')
    @classmethod
    def validate_response_format(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Validate response format."""
        if v and 'type' in v:
            if v['type'] not in ['text', 'json_object']:
                raise ValueError("response_format type must be 'text' or 'json_object'")
        return v


class OllamaLLMConfig(LLMBaseConfig):
    """Configuration for Ollama LLM models."""
    
    model_name: str = Field(
        ...,
        description="Ollama model name"
    )
    base_url: HttpUrl = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate"
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        gt=0,
        description="Context window size"
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream responses"
    )
    request_timeout: float = Field(
        default=120.0,
        gt=0,
        le=600,
        description="Request timeout in seconds"
    )
    num_predict: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of tokens to predict"
    )
    num_ctx: Optional[int] = Field(
        default=None,
        gt=0,
        description="Context size"
    )
    top_k: Optional[int] = Field(
        default=None,
        gt=0,
        description="Top-k sampling"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    repeat_penalty: float = Field(
        default=1.1,
        ge=0.0,
        description="Repetition penalty"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences"
    )
    keep_alive: Optional[str] = Field(
        default=None,
        description="Keep model loaded in memory"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend"
    )
    format: Optional[Literal["json"]] = Field(
        default=None,
        description="Response format"
    )
    raw: bool = Field(
        default=False,
        description="Whether to use raw mode"
    )
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate Ollama model name."""
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()


class PerplexityLLMConfig(APIConfig, LLMBaseConfig):
    """Configuration for Perplexity AI LLM models."""
    
    model_name: str = Field(
        default="llama-3.1-sonar-small-128k-online",
        description="Perplexity model name"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate"
    )
    context_window: int = Field(
        default=127072,
        gt=0,
        description="Context window size"
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream responses"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    top_k: int = Field(
        default=0,
        ge=0,
        description="Top-k sampling (0 = disabled)"
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    frequency_penalty: float = Field(
        default=1.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    return_citations: bool = Field(
        default=False,
        description="Whether to return citations"
    )
    search_domain_filter: Optional[List[str]] = Field(
        default=None,
        description="Domains to search"
    )
    search_recency_filter: Optional[Literal["hour", "day", "week", "month", "year"]] = Field(
        default=None,
        description="Recency filter for search"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend"
    )
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate Perplexity model name."""
        valid_models = [
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-huge-128k-online",
            "llama-3.1-sonar-small-128k-chat",
            "llama-3.1-sonar-large-128k-chat",
            "llama-3.1-8b-instruct",
            "llama-3.1-70b-instruct",
        ]
        if v not in valid_models:
            # Allow custom but could warn
            pass
        return v


class LMStudioLLMConfig(LLMBaseConfig):
    """Configuration for LMStudio LLM models."""
    
    model_name: str = Field(
        ...,
        description="LMStudio model identifier"
    )
    base_url: HttpUrl = Field(
        default="http://localhost:1234/v1",
        description="LMStudio API base URL"
    )
    api_key: SecretStr = Field(
        default=SecretStr("lm-studio"),
        description="API key (usually 'lm-studio')"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate"
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        gt=0,
        description="Context window size"
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream responses"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend"
    )


class OpenRouterLLMConfig(APIConfig, LLMBaseConfig):
    """Configuration for OpenRouter LLM models."""
    
    model_name: str = Field(
        ...,
        description="OpenRouter model identifier"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate"
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        gt=0,
        description="Context window size"
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream responses"
    )
    route: Optional[Literal["fallback"]] = Field(
        default=None,
        description="Routing strategy"
    )
    site_url: Optional[HttpUrl] = Field(
        default=None,
        description="Your site URL for better routing"
    )
    app_name: Optional[str] = Field(
        default=None,
        description="Your app name for identification"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend"
    )
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate OpenRouter model format."""
        # OpenRouter uses provider/model format
        if '/' not in v:
            raise ValueError("OpenRouter model must be in 'provider/model' format")
        return v


class GroqLLMConfig(APIConfig, LLMBaseConfig):
    """Configuration for Groq LLM models."""
    
    model_name: str = Field(
        default="llama-3.1-70b-versatile",
        description="Groq model name"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate"
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        gt=0,
        description="Context window size"
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream responses"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for deterministic generation"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend"
    )
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate Groq model name."""
        valid_models = [
            "llama-3.1-405b-reasoning",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-groq-8b-8192-tool-use-preview",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-11b-text-preview",
            "llama-3.2-90b-text-preview",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "gemma2-9b-it",
        ]
        if v not in valid_models:
            # Allow custom but could warn
            pass
        return v


class TogetherAILLMConfig(APIConfig, LLMBaseConfig):
    """Configuration for Together AI LLM models."""
    
    model_name: str = Field(
        ...,
        description="Together AI model identifier"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate"
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        gt=0,
        description="Context window size"
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream responses"
    )
    top_p: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    top_k: int = Field(
        default=50,
        ge=0,
        description="Top-k sampling"
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=0.0,
        description="Repetition penalty"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend"
    )


class DeepSeekLLMConfig(APIConfig, LLMBaseConfig):
    """Configuration for DeepSeek LLM models."""
    
    model_name: str = Field(
        default="deepseek-chat",
        description="DeepSeek model name"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate"
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        gt=0,
        description="Context window size"
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream responses"
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend"
    )
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate DeepSeek model name."""
        valid_models = ["deepseek-chat", "deepseek-coder"]
        if v not in valid_models:
            # Allow custom but could warn
            pass
        return v


class AnthropicLLMConfig(APIConfig, LLMBaseConfig):
    """Configuration for Anthropic Claude models."""
    
    model_name: str = Field(
        default="claude-3-sonnet-20240229",
        description="Anthropic model name"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (max 1.0 for Claude)"
    )
    max_tokens: int = Field(
        default=1024,
        gt=0,
        le=4096,
        description="Maximum tokens to generate"
    )
    context_window: int = Field(
        default=200000,
        gt=0,
        description="Context window size"
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream responses"
    )
    top_p: float = Field(
        default=0.999,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    top_k: int = Field(
        default=250,
        ge=0,
        description="Top-k sampling"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend"
    )
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate Anthropic model name."""
        valid_prefixes = ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku', 'claude-2']
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            # Allow custom but could warn
            pass
        return v
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Claude has a max temperature of 1.0."""
        if v > 1.0:
            raise ValueError("Claude models have a maximum temperature of 1.0")
        return v


# MLX config for Apple Silicon Macs
class MLXLLMConfig(LLMBaseConfig):
    """Configuration for MLX models on Apple Silicon."""
    
    model_name: str = Field(
        ...,
        description="MLX model identifier or path"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=100,
        gt=0,
        description="Maximum tokens to generate"
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        gt=0,
        description="Context window size"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend"
    )


# Export all configuration classes
__all__ = [
    'OpenAILLMConfig',
    'OllamaLLMConfig',
    'PerplexityLLMConfig',
    'LMStudioLLMConfig',
    'OpenRouterLLMConfig',
    'GroqLLMConfig',
    'TogetherAILLMConfig',
    'DeepSeekLLMConfig',
    'AnthropicLLMConfig',
    'MLXLLMConfig',
]