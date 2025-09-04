"""Backward compatibility layer for rag module.

This module re-exports all public APIs from rag_client package
to maintain backward compatibility with existing code.
"""

# Re-export everything from rag_client
from rag_client import *

# Additional imports that might be needed for backward compatibility
from rag_client.core.workflow import RAGWorkflow
from rag_client.core.models import (
    ChatState,
    QueryState,
    SimpleQueryEngine,
    EmbeddedNode,
    EmbeddedFile,
    Message,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
)
from rag_client.utils.helpers import (
    error,
    clean_special_tokens,
    cache_dir,
    collection_hash,
    list_files,
    read_files,
    convert_str,
    parse_prefixes,
)

# Ensure all names are available
__all__ = [
    # Core classes
    "RAGWorkflow",
    "ChatState",
    "QueryState",
    "SimpleQueryEngine",
    "EmbeddedNode",
    "EmbeddedFile",
    # Request models
    "Message",
    "ChatCompletionRequest",
    "CompletionRequest",
    "EmbeddingRequest",
    # Helper functions
    "error",
    "clean_special_tokens",
    "cache_dir",
    "collection_hash",
    "list_files",
    "read_files",
    "convert_str",
    "parse_prefixes",
    # All other exports from rag_client
    "Config",
    "RetrievalConfig",
    "QueryConfig",
    "ChatConfig",
    "LoggingConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "SplitterConfig",
    "ExtractorConfig",
    "VectorStoreConfig",
    "embedding_model",
    "llm_model",
    # And many more...
]