"""Pydantic schemas for structured RAG client output.

This module provides validated data models for API responses, search results,
and agent-consumable output formats.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class NodeMetadata(BaseModel):
    """Metadata for a retrieved document node."""

    file_name: Optional[str] = Field(None, description="Source file name")
    file_path: Optional[str] = Field(None, description="Full path to source file")
    file_type: Optional[str] = Field(None, description="File extension or MIME type")
    page_number: Optional[int] = Field(None, description="Page number in document")
    section: Optional[str] = Field(None, description="Section heading")
    creation_date: Optional[datetime] = Field(
        None, description="Document creation date"
    )
    last_modified: Optional[datetime] = Field(
        None, description="Last modification date"
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SearchResult(BaseModel):
    """A single search result with score and metadata."""

    text: str = Field(..., description="Retrieved text content")
    score: float = Field(..., description="Relevance score (0-1)")
    node_id: str = Field(..., description="Unique node identifier")
    metadata: NodeMetadata = Field(
        default_factory=NodeMetadata, description="Source metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "RAG combines retrieval with generation...",
                "score": 0.95,
                "node_id": "node_abc123",
                "metadata": {"file_name": "rag_guide.pdf", "page_number": 5},
            }
        }


class SearchResponse(BaseModel):
    """Response from a search operation."""

    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(
        default_factory=list, description="Search results"
    )
    total_results: int = Field(..., description="Total number of results returned")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is RAG?",
                "results": [],
                "total_results": 5,
                "processing_time_ms": 125.5,
            }
        }


class QueryResponse(BaseModel):
    """Response from a query operation with AI-generated answer."""

    query: str = Field(..., description="Original question")
    answer: str = Field(..., description="AI-generated answer")
    sources: List[SearchResult] = Field(
        default_factory=list, description="Source documents used"
    )
    confidence: Optional[float] = Field(None, description="Answer confidence score")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    model: str = Field(..., description="LLM model used")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is RAG?",
                "answer": "RAG (Retrieval-Augmented Generation) is...",
                "sources": [],
                "confidence": 0.85,
                "processing_time_ms": 1250.5,
                "model": "llama3",
            }
        }


class IndexStatus(BaseModel):
    """Status of document indexing."""

    total_documents: int = Field(..., description="Total documents in index")
    total_nodes: int = Field(..., description="Total chunks/nodes in index")
    last_updated: datetime = Field(..., description="Last index update time")
    embedding_model: str = Field(..., description="Embedding model used")
    vector_dimensions: int = Field(..., description="Embedding dimensions")
    index_size_mb: float = Field(..., description="Approximate index size in MB")


class IndexingResult(BaseModel):
    """Result of an indexing operation."""

    success: bool = Field(..., description="Whether indexing succeeded")
    documents_indexed: int = Field(..., description="Number of documents indexed")
    nodes_created: int = Field(..., description="Number of nodes created")
    documents_skipped: int = Field(0, description="Documents skipped (cached)")
    documents_failed: int = Field(0, description="Documents that failed to index")
    failed_files: List[str] = Field(
        default_factory=list, description="Paths of failed files"
    )
    processing_time_ms: float = Field(..., description="Total processing time")
    cache_hit_ratio: float = Field(0.0, description="Ratio of cached vs new documents")


class ErrorResponse(BaseModel):
    """Error response format."""

    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Error timestamp"
    )


class ToolCallType(str, Enum):
    """Types of tool calls for agent integration."""

    SEARCH = "search"
    QUERY = "query"
    INDEX = "index"
    GET_STATUS = "get_status"


class ToolCall(BaseModel):
    """A tool call request from an AI agent."""

    tool_type: ToolCallType = Field(..., description="Type of tool call")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )


class ToolResult(BaseModel):
    """Result of a tool call for agent consumption."""

    tool_type: ToolCallType = Field(..., description="Type of tool that was called")
    success: bool = Field(..., description="Whether the call succeeded")
    result: Union[
        SearchResponse, QueryResponse, IndexingResult, IndexStatus, ErrorResponse
    ] = Field(..., description="Tool-specific result")
    execution_time_ms: float = Field(..., description="Execution time")


class AgentContext(BaseModel):
    """Context information for AI agents.

    Provides structured metadata about RAG capabilities and current state
    for integration with AI agents like Claude, GPT, etc.
    """

    available_tools: List[str] = Field(
        default_factory=lambda: ["search", "query", "index", "get_status"],
        description="Available RAG tools",
    )
    index_status: Optional[IndexStatus] = Field(
        None, description="Current index status"
    )
    capabilities: Dict[str, bool] = Field(
        default_factory=lambda: {
            "search": True,
            "query": True,
            "chat": True,
            "streaming": True,
            "hybrid_search": False,
        },
        description="Available capabilities",
    )
    model_info: Dict[str, str] = Field(
        default_factory=dict, description="Information about configured models"
    )

    class Config:
        json_schema_extra = {
            "description": (
                "Use this context to understand RAG system capabilities. "
                "Call 'search' for document retrieval, 'query' for AI-answered questions."
            )
        }


def search_result_from_node(node: Any, score: float = 0.0) -> SearchResult:
    """Convert a LlamaIndex node to a SearchResult.

    Args:
        node: LlamaIndex NodeWithScore or similar
        score: Relevance score

    Returns:
        Validated SearchResult
    """
    metadata = {}
    if hasattr(node, "metadata"):
        metadata = node.metadata
    elif hasattr(node, "node") and hasattr(node.node, "metadata"):
        metadata = node.node.metadata

    text = ""
    if hasattr(node, "text"):
        text = node.text
    elif hasattr(node, "get_content"):
        text = node.get_content()
    elif hasattr(node, "node"):
        text = (
            node.node.get_content()
            if hasattr(node.node, "get_content")
            else str(node.node)
        )

    node_id = ""
    if hasattr(node, "node_id"):
        node_id = node.node_id
    elif hasattr(node, "id_"):
        node_id = node.id_
    elif hasattr(node, "node") and hasattr(node.node, "id_"):
        node_id = node.node.id_

    if hasattr(node, "score") and node.score is not None:
        score = node.score

    return SearchResult(
        text=text,
        score=score,
        node_id=node_id,
        metadata=NodeMetadata(
            file_name=metadata.get("file_name"),
            file_path=metadata.get("file_path"),
            file_type=metadata.get("file_type"),
            page_number=metadata.get("page_label") or metadata.get("page_number"),
            section=metadata.get("section") or metadata.get("heading"),
            extra={
                k: v
                for k, v in metadata.items()
                if k
                not in (
                    "file_name",
                    "file_path",
                    "file_type",
                    "page_label",
                    "page_number",
                    "section",
                    "heading",
                )
            },
        ),
    )
