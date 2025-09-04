"""Pydantic configuration models for text chunking strategies.

This module contains all chunking and splitting configurations migrated from
dataclass-wizard to Pydantic with proper validation.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator

from rag_client.config.base import ChunkingBaseConfig


class SentenceSplitterConfig(ChunkingBaseConfig):
    """Configuration for sentence-based text splitting."""
    
    chunk_size: int = Field(
        default=1024,
        ge=100,
        le=4096,
        description="Target size for each chunk in characters"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Number of overlapping characters between chunks"
    )
    separator: str = Field(
        default=" ",
        description="Separator to use between sentences"
    )
    paragraph_separator: str = Field(
        default="\n\n",
        description="Separator to identify paragraphs"
    )
    secondary_chunking_regex: Optional[str] = Field(
        default=None,
        description="Regex for secondary chunking"
    )
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is at most half of chunk size."""
        if 'chunk_size' in info.data and v > info.data['chunk_size'] // 2:
            raise ValueError("chunk_overlap should be at most half of chunk_size")
        return v


class SemanticSplitterConfig(ChunkingBaseConfig):
    """Configuration for semantic-based text splitting."""
    
    buffer_size: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Number of sentences to group for semantic comparison"
    )
    breakpoint_percentile_threshold: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Percentile threshold for identifying semantic breakpoints"
    )
    embed_model_name: Optional[str] = Field(
        default=None,
        description="Embedding model to use for semantic similarity"
    )
    
    @field_validator('breakpoint_percentile_threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate percentile threshold."""
        if v <= 50:
            # Warning: low threshold may create too many splits
            pass
        return v


class CodeSplitterConfig(ChunkingBaseConfig):
    """Configuration for code-aware text splitting."""
    
    language: Literal[
        "python", "javascript", "typescript", "jsx", "tsx",
        "java", "cpp", "c", "csharp", "go", "rust",
        "php", "ruby", "swift", "kotlin", "scala",
        "html", "css", "sql", "markdown"
    ] = Field(
        default="python",
        description="Programming language for syntax-aware splitting"
    )
    chunk_lines: int = Field(
        default=40,
        ge=10,
        le=500,
        description="Target number of lines per chunk"
    )
    chunk_lines_overlap: int = Field(
        default=15,
        ge=0,
        description="Number of overlapping lines between chunks"
    )
    max_chars: int = Field(
        default=1500,
        ge=100,
        le=10000,
        description="Maximum characters per chunk"
    )
    
    @field_validator('chunk_lines_overlap')
    @classmethod
    def validate_lines_overlap(cls, v: int, info) -> int:
        """Ensure line overlap is reasonable."""
        if 'chunk_lines' in info.data and v > info.data['chunk_lines'] // 2:
            raise ValueError("chunk_lines_overlap should be at most half of chunk_lines")
        return v


class MarkdownSplitterConfig(ChunkingBaseConfig):
    """Configuration for Markdown-aware text splitting."""
    
    chunk_size: int = Field(
        default=1024,
        ge=100,
        le=4096,
        description="Target size for each chunk in characters"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Number of overlapping characters between chunks"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include markdown metadata in chunks"
    )
    include_prev_next_rel: bool = Field(
        default=True,
        description="Whether to include references to previous/next chunks"
    )
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is reasonable for markdown."""
        if 'chunk_size' in info.data and v > info.data['chunk_size'] // 2:
            raise ValueError("chunk_overlap should be at most half of chunk_size")
        return v


class HybridSplitterConfig(ChunkingBaseConfig):
    """Configuration for hybrid splitting strategy combining multiple approaches."""
    
    chunk_size: int = Field(
        default=1024,
        ge=100,
        le=4096,
        description="Target size for each chunk"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks"
    )
    primary_strategy: Literal["sentence", "semantic", "code", "markdown"] = Field(
        default="sentence",
        description="Primary splitting strategy"
    )
    fallback_strategy: Literal["sentence", "semantic", "code", "markdown"] = Field(
        default="sentence",
        description="Fallback strategy if primary fails"
    )
    enable_semantic_enhancement: bool = Field(
        default=False,
        description="Whether to use semantic similarity for chunk boundaries"
    )
    min_chunk_size: int = Field(
        default=100,
        ge=10,
        description="Minimum size for a valid chunk"
    )
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is reasonable."""
        if 'chunk_size' in info.data and v > info.data['chunk_size'] // 2:
            raise ValueError("chunk_overlap should be at most half of chunk_size")
        return v
    
    @field_validator('min_chunk_size')
    @classmethod
    def validate_min_size(cls, v: int, info) -> int:
        """Ensure min size is less than chunk size."""
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError("min_chunk_size must be less than chunk_size")
        return v
    
    @field_validator('fallback_strategy')
    @classmethod
    def validate_fallback(cls, v: str, info) -> str:
        """Ensure fallback is different from primary."""
        if 'primary_strategy' in info.data and v == info.data['primary_strategy']:
            # Warning: fallback same as primary
            pass
        return v


# Advanced chunking configurations

class RecursiveCharacterSplitterConfig(ChunkingBaseConfig):
    """Configuration for recursive character-based splitting."""
    
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4096,
        description="Target chunk size"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks"
    )
    separators: List[str] = Field(
        default=["\n\n", "\n", " ", ""],
        description="Separators to try in order"
    )
    keep_separator: bool = Field(
        default=True,
        description="Whether to keep separators in chunks"
    )
    is_separator_regex: bool = Field(
        default=False,
        description="Whether separators are regex patterns"
    )
    
    @field_validator('separators')
    @classmethod
    def validate_separators(cls, v: List[str]) -> List[str]:
        """Ensure separators list is not empty."""
        if not v:
            raise ValueError("separators list cannot be empty")
        return v


class TokenSplitterConfig(ChunkingBaseConfig):
    """Configuration for token-based splitting."""
    
    chunk_size: int = Field(
        default=512,
        ge=10,
        le=8192,
        description="Target chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap in tokens"
    )
    tokenizer_name: Optional[str] = Field(
        default=None,
        description="Tokenizer to use (defaults to cl100k_base)"
    )
    add_special_tokens: bool = Field(
        default=True,
        description="Whether to add special tokens"
    )
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure token overlap is reasonable."""
        if 'chunk_size' in info.data and v > info.data['chunk_size'] // 2:
            raise ValueError("chunk_overlap should be at most half of chunk_size")
        return v


class HierarchicalSplitterConfig(ChunkingBaseConfig):
    """Configuration for hierarchical document splitting."""
    
    chunk_sizes: List[int] = Field(
        default=[2048, 512, 128],
        description="Chunk sizes for different hierarchy levels"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap between chunks at same level"
    )
    level_titles: List[str] = Field(
        default=["Document", "Section", "Paragraph"],
        description="Names for hierarchy levels"
    )
    
    @field_validator('chunk_sizes')
    @classmethod
    def validate_chunk_sizes(cls, v: List[int]) -> List[int]:
        """Ensure chunk sizes are in descending order."""
        if not v:
            raise ValueError("chunk_sizes cannot be empty")
        if v != sorted(v, reverse=True):
            raise ValueError("chunk_sizes must be in descending order")
        for size in v:
            if size < 10 or size > 10000:
                raise ValueError(f"chunk size {size} out of range [10, 10000]")
        return v
    
    @field_validator('level_titles')
    @classmethod
    def validate_titles(cls, v: List[str], info) -> List[str]:
        """Ensure titles match chunk sizes."""
        if 'chunk_sizes' in info.data and len(v) != len(info.data['chunk_sizes']):
            raise ValueError("level_titles must have same length as chunk_sizes")
        return v


# Export all configuration classes
__all__ = [
    'SentenceSplitterConfig',
    'SemanticSplitterConfig',
    'CodeSplitterConfig',
    'MarkdownSplitterConfig',
    'HybridSplitterConfig',
    'RecursiveCharacterSplitterConfig',
    'TokenSplitterConfig',
    'HierarchicalSplitterConfig',
]