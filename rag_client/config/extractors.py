"""Pydantic configuration models for content extractors.

This module contains all extractor configurations for entity, keyword,
summary, and QA extraction migrated to Pydantic.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator

from rag_client.config.base import BaseConfig, ModelConfig


class EntityExtractorConfig(ModelConfig):
    """Configuration for entity extraction from documents."""
    
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="LLM model for entity extraction"
    )
    max_entities: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of entities to extract"
    )
    entity_types: List[str] = Field(
        default=["PERSON", "ORGANIZATION", "LOCATION", "DATE", "PRODUCT"],
        description="Types of entities to extract"
    )
    include_confidence: bool = Field(
        default=True,
        description="Whether to include confidence scores"
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    context_window: int = Field(
        default=500,
        gt=0,
        le=2000,
        description="Context window around entity mentions"
    )
    deduplicate: bool = Field(
        default=True,
        description="Whether to deduplicate entities"
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether entity matching is case-sensitive"
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for entity extraction"
    )
    
    @field_validator('entity_types')
    @classmethod
    def validate_entity_types(cls, v: List[str]) -> List[str]:
        """Ensure entity types are not empty."""
        if not v:
            raise ValueError("entity_types cannot be empty")
        # Normalize to uppercase
        return [t.upper() for t in v]
    
    @field_validator('max_entities')
    @classmethod
    def validate_max_entities(cls, v: int) -> int:
        """Validate reasonable entity limits."""
        if v > 50:
            # Warning: extracting many entities may be slow
            pass
        return v


class KeywordExtractorConfig(ModelConfig):
    """Configuration for keyword extraction from documents."""
    
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="LLM model for keyword extraction"
    )
    keyword_count: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of keywords to extract"
    )
    extraction_method: Literal["llm", "tfidf", "rake", "yake", "keybert"] = Field(
        default="llm",
        description="Keyword extraction method"
    )
    include_scores: bool = Field(
        default=True,
        description="Whether to include relevance scores"
    )
    min_keyword_length: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Minimum keyword length in characters"
    )
    max_keyword_length: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Maximum keyword length in characters"
    )
    ngram_range: tuple[int, int] = Field(
        default=(1, 3),
        description="N-gram range for multi-word keywords"
    )
    remove_stopwords: bool = Field(
        default=True,
        description="Whether to remove stopwords"
    )
    language: str = Field(
        default="english",
        description="Language for stopword removal"
    )
    custom_stopwords: Optional[List[str]] = Field(
        default=None,
        description="Additional stopwords to exclude"
    )
    diversity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Diversity of keywords (for KeyBERT)"
    )
    
    @field_validator('max_keyword_length')
    @classmethod
    def validate_keyword_lengths(cls, v: int, info) -> int:
        """Ensure max length is greater than min length."""
        if 'min_keyword_length' in info.data and v <= info.data['min_keyword_length']:
            raise ValueError("max_keyword_length must be greater than min_keyword_length")
        return v
    
    @field_validator('ngram_range')
    @classmethod
    def validate_ngram_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Validate n-gram range."""
        if len(v) != 2:
            raise ValueError("ngram_range must be a tuple of (min, max)")
        if v[0] < 1 or v[1] < v[0] or v[1] > 5:
            raise ValueError("Invalid ngram_range: must be 1 <= min <= max <= 5")
        return v


class SummaryExtractorConfig(ModelConfig):
    """Configuration for summary generation from documents."""
    
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="LLM model for summarization"
    )
    summary_length: int = Field(
        default=150,
        ge=20,
        le=1000,
        description="Target summary length in words"
    )
    summary_type: Literal["abstractive", "extractive", "hybrid"] = Field(
        default="abstractive",
        description="Type of summarization"
    )
    style: Literal["paragraph", "bullets", "numbered"] = Field(
        default="paragraph",
        description="Summary output style"
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific aspects to focus on"
    )
    preserve_key_phrases: bool = Field(
        default=True,
        description="Whether to preserve important phrases"
    )
    include_metadata: bool = Field(
        default=False,
        description="Whether to include document metadata"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for summary generation"
    )
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Custom instructions for summarization"
    )
    chunk_summaries: bool = Field(
        default=False,
        description="Whether to summarize chunks separately first"
    )
    max_chunk_size: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum chunk size for chunked summarization"
    )
    
    @field_validator('summary_length')
    @classmethod
    def validate_summary_length(cls, v: int, info) -> int:
        """Validate summary length based on type."""
        summary_type = info.data.get('summary_type', 'abstractive')
        if summary_type == 'extractive' and v > 500:
            # Warning: extractive summaries work better with shorter lengths
            pass
        return v


class QAExtractorConfig(ModelConfig):
    """Configuration for question-answer pair extraction from documents."""
    
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="LLM model for QA generation"
    )
    num_questions: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Number of QA pairs to generate"
    )
    question_types: List[Literal["factual", "conceptual", "analytical", "comparative"]] = Field(
        default=["factual", "conceptual"],
        description="Types of questions to generate"
    )
    answer_style: Literal["brief", "detailed", "academic"] = Field(
        default="brief",
        description="Style of answers"
    )
    max_answer_length: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum answer length in words"
    )
    include_context: bool = Field(
        default=True,
        description="Whether to include source context"
    )
    difficulty_level: Literal["basic", "intermediate", "advanced"] = Field(
        default="intermediate",
        description="Difficulty level of questions"
    )
    avoid_yes_no: bool = Field(
        default=True,
        description="Whether to avoid yes/no questions"
    )
    custom_guidelines: Optional[str] = Field(
        default=None,
        description="Custom guidelines for QA generation"
    )
    validation_model: Optional[str] = Field(
        default=None,
        description="Model for validating QA pairs"
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for QA pairs"
    )
    
    @field_validator('question_types')
    @classmethod
    def validate_question_types(cls, v: List[str]) -> List[str]:
        """Ensure question types are not empty."""
        if not v:
            raise ValueError("question_types cannot be empty")
        return v
    
    @field_validator('num_questions')
    @classmethod
    def validate_num_questions(cls, v: int, info) -> int:
        """Validate question count based on difficulty."""
        difficulty = info.data.get('difficulty_level', 'intermediate')
        if difficulty == 'advanced' and v > 15:
            # Warning: generating many advanced questions may be slow
            pass
        return v


class MetadataExtractorConfig(BaseConfig):
    """Configuration for metadata extraction from documents."""
    
    extract_title: bool = Field(
        default=True,
        description="Whether to extract document title"
    )
    extract_author: bool = Field(
        default=True,
        description="Whether to extract author information"
    )
    extract_date: bool = Field(
        default=True,
        description="Whether to extract dates"
    )
    extract_language: bool = Field(
        default=True,
        description="Whether to detect language"
    )
    extract_categories: bool = Field(
        default=True,
        description="Whether to extract categories/tags"
    )
    max_categories: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of categories"
    )
    custom_fields: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom metadata fields to extract"
    )
    date_formats: List[str] = Field(
        default=["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
        description="Date formats to recognize"
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for extraction"
    )


class TableExtractorConfig(BaseConfig):
    """Configuration for table extraction from documents."""
    
    extract_format: Literal["markdown", "csv", "json", "html"] = Field(
        default="markdown",
        description="Output format for extracted tables"
    )
    preserve_formatting: bool = Field(
        default=True,
        description="Whether to preserve table formatting"
    )
    merge_cells: bool = Field(
        default=True,
        description="Whether to handle merged cells"
    )
    extract_headers: bool = Field(
        default=True,
        description="Whether to identify table headers"
    )
    min_rows: int = Field(
        default=2,
        ge=1,
        description="Minimum rows for valid table"
    )
    min_columns: int = Field(
        default=2,
        ge=1,
        description="Minimum columns for valid table"
    )
    clean_data: bool = Field(
        default=True,
        description="Whether to clean extracted data"
    )
    include_caption: bool = Field(
        default=True,
        description="Whether to extract table captions"
    )


# Export all configuration classes
__all__ = [
    'EntityExtractorConfig',
    'KeywordExtractorConfig',
    'SummaryExtractorConfig',
    'QAExtractorConfig',
    'MetadataExtractorConfig',
    'TableExtractorConfig',
]