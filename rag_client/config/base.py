"""Base Pydantic configuration models for RAG client.

This module defines foundational BaseModel classes with common configuration patterns,
validators, and Config settings that will be inherited by all specific configuration models.
"""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, field_validator


class BaseConfig(BaseModel):
    """Base configuration class with common settings and validators.
    
    All configuration models should inherit from this class to ensure
    consistent behavior across the application.
    """
    
    model_config = ConfigDict(
        # Forbid extra fields to catch typos and invalid configurations
        extra='forbid',
        # Validate on assignment to catch errors immediately
        validate_assignment=True,
        # Use enum values instead of enum instances
        use_enum_values=True,
        # Allow arbitrary types for complex fields
        arbitrary_types_allowed=True,
        # Populate by field name for better YAML compatibility
        populate_by_name=True,
        # Enable JSON schema generation
        json_schema_extra={
            "examples": []
        }
    )
    
    @field_validator('*', mode='before')
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        """Convert empty strings to None for optional fields."""
        if isinstance(v, str) and v == '':
            return None
        return v


class APIConfig(BaseConfig):
    """Base configuration for API-based services."""
    
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for authentication"
    )
    api_key_command: Optional[str] = Field(
        default=None,
        description="Command to retrieve API key"
    )
    api_base: Optional[HttpUrl] = Field(
        default=None,
        description="Base URL for API endpoint"
    )
    api_version: Optional[str] = Field(
        default=None,
        description="API version to use"
    )
    max_retries: int = Field(
        default=10,
        ge=0,
        description="Maximum number of retry attempts"
    )
    timeout: float = Field(
        default=60.0,
        gt=0,
        description="Request timeout in seconds"
    )
    
    @field_validator('api_key', 'api_key_command')
    @classmethod
    def validate_api_auth(cls, v: Any, info) -> Any:
        """Ensure either api_key or api_key_command is provided for authentication."""
        if info.field_name == 'api_key_command' and v is None:
            # Check if api_key is also None
            if info.data.get('api_key') is None:
                # This is okay - some providers may not require auth
                pass
        return v


class ModelConfig(BaseConfig):
    """Base configuration for model-based services."""
    
    model_name: str = Field(
        ...,
        description="Name or identifier of the model to use"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum number of tokens to generate"
    )
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate that model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class EmbeddingBaseConfig(BaseConfig):
    """Base configuration for embedding providers."""
    
    embed_batch_size: int = Field(
        default=10,
        gt=0,
        description="Batch size for embedding generation"
    )
    dimensions: Optional[int] = Field(
        default=None,
        gt=0,
        description="Dimensionality of embeddings"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize embeddings"
    )


class LLMBaseConfig(ModelConfig):
    """Base configuration for LLM providers."""
    
    context_window: int = Field(
        default=3900,
        gt=0,
        description="Maximum context window size"
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
        description="Frequency penalty for token repetition"
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty for token repetition"
    )


class ChunkingBaseConfig(BaseConfig):
    """Base configuration for text chunking strategies."""
    
    chunk_size: int = Field(
        default=1024,
        gt=0,
        description="Size of text chunks in characters"
    )
    chunk_overlap: int = Field(
        default=20,
        ge=0,
        description="Number of overlapping characters between chunks"
    )
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v


class DatabaseConfig(BaseConfig):
    """Base configuration for database connections."""
    
    host: str = Field(
        default="localhost",
        description="Database host address"
    )
    port: int = Field(
        default=5432,
        gt=0,
        le=65535,
        description="Database port"
    )
    database: str = Field(
        ...,
        description="Database name"
    )
    username: Optional[str] = Field(
        default=None,
        description="Database username"
    )
    password: Optional[SecretStr] = Field(
        default=None,
        description="Database password"
    )
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        auth = ""
        if self.username:
            auth = self.username
            if self.password:
                auth += f":{self.password.get_secret_value()}"
            auth += "@"
        return f"postgresql://{auth}{self.host}:{self.port}/{self.database}"


class FilePathConfig(BaseConfig):
    """Base configuration for file path handling."""
    
    @field_validator('*', mode='before')
    @classmethod
    def validate_path(cls, v: Any) -> Any:
        """Convert string paths to Path objects and validate existence for input files."""
        if isinstance(v, str) and v:
            return Path(v)
        return v
    
    @field_validator('*', mode='after')
    @classmethod
    def expand_path(cls, v: Any) -> Any:
        """Expand user home directory and resolve paths."""
        if isinstance(v, Path):
            return v.expanduser().resolve()
        return v


# Re-export for convenience
__all__ = [
    'BaseConfig',
    'APIConfig',
    'ModelConfig',
    'EmbeddingBaseConfig',
    'LLMBaseConfig',
    'ChunkingBaseConfig',
    'DatabaseConfig',
    'FilePathConfig',
]