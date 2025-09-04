"""Error handling and response models for the FastAPI server.

This module provides consistent error response formatting and
exception mapping for the API layer.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..exceptions import (
    APIError,
    ConfigurationError,
    DocumentProcessingError,
    IndexingError,
    ProviderError,
    RAGClientError,
    RateLimitError,
    RetrievalError,
    StorageError,
    TimeoutError,
    ValidationError,
)

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Human-readable error message")
    type: str = Field(..., description="Error type identifier")


class ErrorResponse(BaseModel):
    """Standard error response format for API."""
    
    error: str = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[list[ErrorDetail]] = Field(None, description="Additional error details")
    correlation_id: str = Field(..., description="Unique identifier for this error instance")
    timestamp: str = Field(..., description="ISO 8601 timestamp when error occurred")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Invalid request parameters",
                "details": [
                    {
                        "field": "temperature",
                        "message": "Value must be between 0 and 2",
                        "type": "range_error"
                    }
                ],
                "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2024-01-15T10:30:00Z",
                "context": {
                    "endpoint": "/v1/chat/completions",
                    "method": "POST"
                }
            }
        }


# Exception to HTTP status code mapping
EXCEPTION_STATUS_MAP = {
    ConfigurationError: status.HTTP_400_BAD_REQUEST,
    ValidationError: status.HTTP_400_BAD_REQUEST,
    APIError: status.HTTP_400_BAD_REQUEST,
    DocumentProcessingError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    IndexingError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    RetrievalError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    StorageError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ProviderError: status.HTTP_502_BAD_GATEWAY,
    TimeoutError: status.HTTP_504_GATEWAY_TIMEOUT,
    RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
}


def create_error_response(
    exception: Exception,
    request: Request,
    correlation_id: Optional[str] = None
) -> ErrorResponse:
    """Create a standardized error response from an exception.
    
    Args:
        exception: The exception that occurred
        request: The FastAPI request object
        correlation_id: Optional correlation ID (generated if not provided)
        
    Returns:
        ErrorResponse object with formatted error information
    """
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Build context
    context = {
        "endpoint": str(request.url.path),
        "method": request.method
    }
    
    # Handle RAGClientError and its subclasses
    if isinstance(exception, RAGClientError):
        error_code = exception.error_code
        message = exception.message
        
        # Add exception context to response context
        if exception.context:
            context.update(exception.context)
        
        # Create error details if available
        details = None
        if hasattr(exception, 'field') and exception.field:
            details = [
                ErrorDetail(
                    field=exception.field,
                    message=message,
                    type=exception.__class__.__name__
                )
            ]
    else:
        # Handle unexpected exceptions
        error_code = "INTERNAL_ERROR"
        message = "An unexpected error occurred"
        details = None
        
        # Log the full exception for debugging
        logger.error(
            f"Unexpected error in {request.method} {request.url.path}",
            exc_info=exception,
            extra={"correlation_id": correlation_id}
        )
    
    return ErrorResponse(
        error=error_code,
        message=message,
        details=details,
        correlation_id=correlation_id,
        timestamp=timestamp,
        context=context
    )


def get_status_code(exception: Exception) -> int:
    """Get the appropriate HTTP status code for an exception.
    
    Args:
        exception: The exception to map
        
    Returns:
        HTTP status code
    """
    # Check direct exception type
    for exc_type, status_code in EXCEPTION_STATUS_MAP.items():
        if isinstance(exception, exc_type):
            return status_code
    
    # Check if it's an APIError with explicit status code
    if isinstance(exception, APIError) and exception.context.get('status_code'):
        return exception.context['status_code']
    
    # Default to 500 for unexpected errors
    return status.HTTP_500_INTERNAL_SERVER_ERROR


async def rag_exception_handler(request: Request, exc: RAGClientError) -> JSONResponse:
    """Global exception handler for RAGClientError and subclasses.
    
    Args:
        request: The FastAPI request that caused the exception
        exc: The RAGClientError exception
        
    Returns:
        JSONResponse with error details
    """
    correlation_id = str(uuid.uuid4())
    
    # Log the error with correlation ID
    logger.error(
        f"RAGClientError in {request.method} {request.url.path}: {exc}",
        extra={
            "correlation_id": correlation_id,
            "error_code": exc.error_code,
            "context": exc.context
        }
    )
    
    # Create error response
    error_response = create_error_response(exc, request, correlation_id)
    
    # Get appropriate status code
    status_code = get_status_code(exc)
    
    # Add retry-after header for rate limit errors
    headers = {}
    if isinstance(exc, RateLimitError) and exc.context.get('retry_after'):
        headers['Retry-After'] = str(exc.context['retry_after'])
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(exclude_none=True),
        headers=headers
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unexpected exceptions.
    
    Args:
        request: The FastAPI request that caused the exception
        exc: The unexpected exception
        
    Returns:
        JSONResponse with error details
    """
    correlation_id = str(uuid.uuid4())
    
    # Log the full exception
    logger.exception(
        f"Unexpected error in {request.method} {request.url.path}",
        extra={"correlation_id": correlation_id}
    )
    
    # Create generic error response
    error_response = ErrorResponse(
        error="INTERNAL_ERROR",
        message="An internal server error occurred",
        correlation_id=correlation_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        context={
            "endpoint": str(request.url.path),
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(exclude_none=True)
    )


def setup_exception_handlers(app):
    """Setup exception handlers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Register RAGClientError handler
    app.add_exception_handler(RAGClientError, rag_exception_handler)
    
    # Register handlers for specific exceptions
    for exc_class in [
        ConfigurationError,
        IndexingError,
        RetrievalError,
        StorageError,
        ProviderError,
        DocumentProcessingError,
        APIError,
        ValidationError,
        TimeoutError,
        RateLimitError
    ]:
        app.add_exception_handler(exc_class, rag_exception_handler)
    
    # Register general exception handler
    app.add_exception_handler(Exception, general_exception_handler)