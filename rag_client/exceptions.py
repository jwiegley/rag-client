"""Custom exception hierarchy for RAG client.

This module defines custom exceptions with proper inheritance structure
for better error handling and debugging throughout the application.
"""

from typing import Any, Dict, Optional


class RAGClientError(Exception):
    """Base exception class for all RAG client errors.
    
    All custom exceptions should inherit from this class to provide
    a consistent error handling interface.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize RAGClientError.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            context: Optional context data for debugging
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause
        
        # Chain the cause if provided
        if cause:
            self.__cause__ = cause
    
    def __str__(self) -> str:
        """Return formatted error message."""
        parts = [f"[{self.error_code}] {self.message}"]
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        
        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {str(self.cause)}")
        
        return " | ".join(parts)
    
    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"context={self.context!r}, "
            f"cause={self.cause!r})"
        )


class ConfigurationError(RAGClientError):
    """Raised when there's an error in configuration.
    
    This includes invalid YAML files, missing required fields,
    type mismatches, or invalid values.
    """
    
    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        field: Optional[str] = None,
        **kwargs
    ):
        """Initialize ConfigurationError.
        
        Args:
            message: Error description
            config_file: Path to the configuration file
            field: Specific configuration field that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if config_file:
            context['config_file'] = config_file
        if field:
            context['field'] = field
        
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            context=context,
            **kwargs
        )


class IndexingError(RAGClientError):
    """Raised when there's an error during document indexing.
    
    This includes failures in document parsing, embedding generation,
    or storing vectors in the database.
    """
    
    def __init__(
        self,
        message: str,
        document: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        """Initialize IndexingError.
        
        Args:
            message: Error description
            document: Document being indexed when error occurred
            operation: Specific indexing operation that failed
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if document:
            context['document'] = document
        if operation:
            context['operation'] = operation
        
        super().__init__(
            message=message,
            error_code="INDEX_ERROR",
            context=context,
            **kwargs
        )


class RetrievalError(RAGClientError):
    """Raised when there's an error during document retrieval.
    
    This includes failures in query processing, vector search,
    or result ranking.
    """
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        retriever_type: Optional[str] = None,
        **kwargs
    ):
        """Initialize RetrievalError.
        
        Args:
            message: Error description
            query: Query that caused the error
            retriever_type: Type of retriever that failed
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if query:
            context['query'] = query
        if retriever_type:
            context['retriever_type'] = retriever_type
        
        super().__init__(
            message=message,
            error_code="RETRIEVAL_ERROR",
            context=context,
            **kwargs
        )


class StorageError(RAGClientError):
    """Raised when there's an error with storage operations.
    
    This includes database connection failures, query errors,
    or file system issues.
    """
    
    def __init__(
        self,
        message: str,
        storage_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        """Initialize StorageError.
        
        Args:
            message: Error description
            storage_type: Type of storage (e.g., 'postgres', 'file')
            operation: Storage operation that failed
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if storage_type:
            context['storage_type'] = storage_type
        if operation:
            context['operation'] = operation
        
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            context=context,
            **kwargs
        )


class ProviderError(RAGClientError):
    """Raised when there's an error with external providers.
    
    This includes failures with embedding providers, LLM providers,
    or other external services.
    """
    
    def __init__(
        self,
        message: str,
        provider_type: Optional[str] = None,
        provider_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize ProviderError.
        
        Args:
            message: Error description
            provider_type: Type of provider (e.g., 'embedding', 'llm')
            provider_name: Name of the provider (e.g., 'openai', 'huggingface')
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if provider_type:
            context['provider_type'] = provider_type
        if provider_name:
            context['provider_name'] = provider_name
        
        super().__init__(
            message=message,
            error_code="PROVIDER_ERROR",
            context=context,
            **kwargs
        )


class DocumentProcessingError(RAGClientError):
    """Raised when there's an error processing documents.
    
    This includes failures in parsing PDFs, extracting text,
    chunking documents, or handling different file formats.
    """
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        processing_stage: Optional[str] = None,
        **kwargs
    ):
        """Initialize DocumentProcessingError.
        
        Args:
            message: Error description
            file_path: Path to the document being processed
            file_type: Type of document (e.g., 'pdf', 'txt', 'org')
            processing_stage: Stage where processing failed
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if file_path:
            context['file_path'] = file_path
        if file_type:
            context['file_type'] = file_type
        if processing_stage:
            context['processing_stage'] = processing_stage
        
        super().__init__(
            message=message,
            error_code="DOC_PROCESSING_ERROR",
            context=context,
            **kwargs
        )


class APIError(RAGClientError):
    """Raised when there's an error in the API layer.
    
    This includes invalid requests, authentication failures,
    or internal server errors.
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        **kwargs
    ):
        """Initialize APIError.
        
        Args:
            message: Error description
            status_code: HTTP status code
            endpoint: API endpoint that failed
            method: HTTP method used
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if status_code:
            context['status_code'] = status_code
        if endpoint:
            context['endpoint'] = endpoint
        if method:
            context['method'] = method
        
        # Set appropriate error code based on status code
        if status_code:
            if status_code == 400:
                error_code = "BAD_REQUEST"
            elif status_code == 401:
                error_code = "UNAUTHORIZED"
            elif status_code == 403:
                error_code = "FORBIDDEN"
            elif status_code == 404:
                error_code = "NOT_FOUND"
            elif status_code == 429:
                error_code = "RATE_LIMITED"
            elif 500 <= status_code < 600:
                error_code = "SERVER_ERROR"
            else:
                error_code = "API_ERROR"
        else:
            error_code = "API_ERROR"
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            **kwargs
        )


class ValidationError(RAGClientError):
    """Raised when input validation fails.
    
    This includes invalid parameters, out-of-range values,
    or type mismatches.
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        **kwargs
    ):
        """Initialize ValidationError.
        
        Args:
            message: Error description
            field: Field that failed validation
            value: Invalid value provided
            expected_type: Expected type or format
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if field:
            context['field'] = field
        if value is not None:
            context['value'] = str(value)
        if expected_type:
            context['expected_type'] = expected_type
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            context=context,
            **kwargs
        )


class TimeoutError(RAGClientError):
    """Raised when an operation times out.
    
    This includes API timeouts, database query timeouts,
    or long-running operations.
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        """Initialize TimeoutError.
        
        Args:
            message: Error description
            operation: Operation that timed out
            timeout_seconds: Timeout duration in seconds
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if operation:
            context['operation'] = operation
        if timeout_seconds:
            context['timeout_seconds'] = timeout_seconds
        
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            context=context,
            **kwargs
        )


class RateLimitError(ProviderError):
    """Raised when a rate limit is exceeded.
    
    This is a specialized provider error for rate limiting scenarios.
    """
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """Initialize RateLimitError.
        
        Args:
            message: Error description
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if retry_after:
            context['retry_after'] = retry_after
        
        kwargs['context'] = context
        kwargs['error_code'] = "RATE_LIMIT_ERROR"
        
        super().__init__(message=message, **kwargs)


class EmbeddingError(ProviderError):
    """Raised when there's an error generating embeddings.
    
    This includes model initialization failures, invalid inputs,
    or embedding dimension mismatches.
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize EmbeddingError.
        
        Args:
            message: Error description
            model_name: Name of the embedding model
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if model_name:
            context['model_name'] = model_name
        
        kwargs['context'] = context
        kwargs['provider_type'] = 'embedding'
        
        super().__init__(message=message, **kwargs)


class LLMError(ProviderError):
    """Raised when there's an error with LLM operations.
    
    This includes model initialization failures, generation errors,
    or context length exceeded errors.
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize LLMError.
        
        Args:
            message: Error description
            model_name: Name of the LLM model
            **kwargs: Additional arguments passed to parent class
        """
        context = kwargs.pop('context', {})
        if model_name:
            context['model_name'] = model_name
        
        kwargs['context'] = context
        kwargs['provider_type'] = 'llm'
        
        super().__init__(message=message, **kwargs)


# Re-export all exception classes
__all__ = [
    'RAGClientError',
    'ConfigurationError',
    'IndexingError',
    'RetrievalError',
    'StorageError',
    'ProviderError',
    'DocumentProcessingError',
    'APIError',
    'ValidationError',
    'TimeoutError',
    'RateLimitError',
    'EmbeddingError',
    'LLMError',
]