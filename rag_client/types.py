"""Type definitions and aliases for RAG client.

This module provides centralized type definitions, aliases, and protocols
used throughout the RAG client application.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import Literal, NotRequired, TypeAlias, TypedDict

# Import llama-index types when available
if TYPE_CHECKING:
    import numpy as np
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.base.llms.base import BaseLLM
    from llama_index.core.callbacks.base import CallbackManager
    from llama_index.core.chat_engine.types import BaseChatEngine
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core.llms.llm import LLM
    from llama_index.core.response_synthesizers import BaseSynthesizer
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import (
        BaseNode,
        Document,
        MetadataMode,
        NodeWithScore,
        QueryBundle,
        TextNode,
    )
    from llama_index.core.service_context import ServiceContext
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.vector_stores.types import VectorStore
    from numpy.typing import NDArray


# Basic type aliases
NodeID: TypeAlias = str
DocumentID: TypeAlias = str
ChunkID: TypeAlias = str
QueryID: TypeAlias = str
SessionID: TypeAlias = str

# Embedding types
Embedding: TypeAlias = List[float]
EmbeddingVector: TypeAlias = Union[List[float], "NDArray[np.float32]"]
EmbeddingList: TypeAlias = List[Embedding]
EmbeddingDict: TypeAlias = Dict[str, Embedding]

# Document types
DocumentDict: TypeAlias = Dict[str, Any]
DocumentList: TypeAlias = List["Document"]
DocumentMetadata: TypeAlias = Dict[str, Any]
DocumentChunk: TypeAlias = Dict[str, Union[str, int, DocumentMetadata]]

# Node types
NodeList: TypeAlias = List["BaseNode"]
NodeDict: TypeAlias = Dict[NodeID, "BaseNode"]
ScoredNode: TypeAlias = Tuple["BaseNode", float]
ScoredNodeList: TypeAlias = List["NodeWithScore"]

# Query types
QueryResult: TypeAlias = Dict[str, Any]
QueryResponse: TypeAlias = Union[str, Dict[str, Any]]
SearchResult: TypeAlias = Dict[str, Union[str, float, DocumentMetadata]]
SearchResults: TypeAlias = List[SearchResult]

# Configuration types
ConfigDict: TypeAlias = Dict[str, Any]
ProviderConfig: TypeAlias = Dict[str, Union[str, int, float, bool, None]]
ModelConfig: TypeAlias = Dict[str, Any]

# LLM types
LLMResponse: TypeAlias = Union[str, Dict[str, Any]]
ChatMessage: TypeAlias = Dict[Literal["role", "content"], str]
ChatHistory: TypeAlias = List[ChatMessage]
StreamResponse: TypeAlias = Iterator[str]
AsyncStreamResponse: TypeAlias = AsyncIterator[str]

# Storage types
StorageConfig: TypeAlias = Dict[str, Any]
VectorStoreConfig: TypeAlias = Dict[str, Any]
IndexConfig: TypeAlias = Dict[str, Any]

# API types
HTTPMethod: TypeAlias = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
StatusCode: TypeAlias = int
Headers: TypeAlias = Dict[str, str]
RequestBody: TypeAlias = Dict[str, Any]
ResponseBody: TypeAlias = Dict[str, Any]

# Callback types
CallbackFn: TypeAlias = Callable[[Any], None]
AsyncCallbackFn: TypeAlias = Callable[[Any], Awaitable[None]]
EventCallback: TypeAlias = Callable[[str, Dict[str, Any]], None]

# Generic type variables
T = TypeVar('T')
TConfig = TypeVar('TConfig', bound='BaseConfig')
TProvider = TypeVar('TProvider', bound='BaseProvider')
TIndex = TypeVar('TIndex', bound='BaseIndex')


# Typed dictionaries for structured data
class DocumentData(TypedDict):
    """Typed dictionary for document data."""
    text: str
    metadata: NotRequired[DocumentMetadata]
    doc_id: NotRequired[DocumentID]
    embedding: NotRequired[Embedding]


class QueryData(TypedDict):
    """Typed dictionary for query data."""
    query_str: str
    top_k: NotRequired[int]
    filters: NotRequired[Dict[str, Any]]
    alpha: NotRequired[float]


class ChatCompletionMessage(TypedDict):
    """Typed dictionary for chat completion messages."""
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: NotRequired[str]
    function_call: NotRequired[Dict[str, Any]]


class ChatCompletionRequest(TypedDict):
    """Typed dictionary for chat completion requests."""
    messages: List[ChatCompletionMessage]
    model: str
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    stream: NotRequired[bool]
    top_p: NotRequired[float]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]


# Protocol definitions for duck typing
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def get_text_embedding(self, text: str) -> Embedding:
        """Get embedding for a single text."""
        ...
    
    def get_text_embedding_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> EmbeddingList:
        """Get embeddings for multiple texts."""
        ...
    
    @property
    def embed_batch_size(self) -> int:
        """Batch size for embedding generation."""
        ...


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    def complete(
        self,
        prompt: str,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate completion for a prompt."""
        ...
    
    def chat(
        self,
        messages: ChatHistory,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate chat completion."""
        ...
    
    def stream_complete(
        self,
        prompt: str,
        **kwargs: Any
    ) -> StreamResponse:
        """Stream completion for a prompt."""
        ...
    
    def stream_chat(
        self,
        messages: ChatHistory,
        **kwargs: Any
    ) -> StreamResponse:
        """Stream chat completion."""
        ...
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Provider metadata."""
        ...


class StorageBackend(Protocol):
    """Protocol for storage backends."""
    
    def add_documents(self, documents: DocumentList) -> List[DocumentID]:
        """Add documents to storage."""
        ...
    
    def delete_document(self, doc_id: DocumentID) -> bool:
        """Delete a document from storage."""
        ...
    
    def get_document(self, doc_id: DocumentID) -> Optional["Document"]:
        """Retrieve a document by ID."""
        ...
    
    def query(
        self,
        query_embedding: Embedding,
        top_k: int = 10,
        **kwargs: Any
    ) -> ScoredNodeList:
        """Query storage with embedding."""
        ...


class Retriever(Protocol):
    """Protocol for retrievers."""
    
    def retrieve(
        self,
        query_str: str,
        top_k: int = 10
    ) -> ScoredNodeList:
        """Retrieve relevant nodes for a query."""
        ...
    
    async def aretrieve(
        self,
        query_str: str,
        top_k: int = 10
    ) -> ScoredNodeList:
        """Asynchronously retrieve relevant nodes."""
        ...


class BaseProvider(ABC):
    """Abstract base class for providers."""
    
    @abstractmethod
    def initialize(self, config: ProviderConfig) -> None:
        """Initialize the provider with configuration."""
        pass
    
    @abstractmethod
    def validate_config(self, config: ProviderConfig) -> bool:
        """Validate provider configuration."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


# Utility type guards
def is_embedding_list(obj: Any) -> bool:
    """Check if object is a list of embeddings."""
    return (
        isinstance(obj, list) and
        all(isinstance(item, list) and 
            all(isinstance(x, (int, float)) for x in item)
            for item in obj)
    )


def is_document_list(obj: Any) -> bool:
    """Check if object is a list of documents."""
    if not isinstance(obj, list):
        return False
    
    # Import here to avoid circular imports
    from llama_index.core.schema import Document
    return all(isinstance(item, Document) for item in obj)


def is_chat_history(obj: Any) -> bool:
    """Check if object is a valid chat history."""
    if not isinstance(obj, list):
        return False
    
    return all(
        isinstance(msg, dict) and
        "role" in msg and
        "content" in msg and
        msg["role"] in ["system", "user", "assistant", "function"]
        for msg in obj
    )


# Re-export commonly used types
__all__ = [
    # Type aliases
    'NodeID',
    'DocumentID',
    'ChunkID',
    'QueryID',
    'SessionID',
    'Embedding',
    'EmbeddingVector',
    'EmbeddingList',
    'EmbeddingDict',
    'DocumentDict',
    'DocumentList',
    'DocumentMetadata',
    'DocumentChunk',
    'NodeList',
    'NodeDict',
    'ScoredNode',
    'ScoredNodeList',
    'QueryResult',
    'QueryResponse',
    'SearchResult',
    'SearchResults',
    'ConfigDict',
    'ProviderConfig',
    'ModelConfig',
    'LLMResponse',
    'ChatMessage',
    'ChatHistory',
    'StreamResponse',
    'AsyncStreamResponse',
    'StorageConfig',
    'VectorStoreConfig',
    'IndexConfig',
    'HTTPMethod',
    'StatusCode',
    'Headers',
    'RequestBody',
    'ResponseBody',
    'CallbackFn',
    'AsyncCallbackFn',
    'EventCallback',
    # Type variables
    'T',
    'TConfig',
    'TProvider',
    'TIndex',
    # Typed dictionaries
    'DocumentData',
    'QueryData',
    'ChatCompletionMessage',
    'ChatCompletionRequest',
    # Protocols
    'EmbeddingProvider',
    'LLMProvider',
    'StorageBackend',
    'Retriever',
    'BaseProvider',
    # Type guards
    'is_embedding_list',
    'is_document_list',
    'is_chat_history',
]