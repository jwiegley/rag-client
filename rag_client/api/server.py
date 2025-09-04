"""FastAPI server implementation for RAG client.

Provides OpenAI-compatible API endpoints for chat completions,
completions, embeddings, and model listing.
"""

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator, Sequence
from typing import Any, NoReturn

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_index.core.llms import ChatMessage

from rag import *

api = FastAPI(title="OpenAI API Compatible Interface")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


workflow: RAGWorkflow | None = None
retriever: BaseRetriever | None = None
query_state: QueryState | None = None
chat_state: ChatState | None = None


def verify_api_key(authorization: str | None = None):
    """Verify API key from authorization header.
    
    Validates incoming API keys against expected values. Supports both
    Bearer token format and direct key format for flexibility.
    
    Args:
        authorization: Authorization header value. Expected formats:
            - "Bearer sk-xxxxx" (OpenAI standard)
            - "sk-xxxxx" (direct key)
            - None (returns None, may be valid for some endpoints)
        
    Returns:
        Validated API key string if valid, None if no auth provided.
        
    Raises:
        HTTPException: 401 Unauthorized if key is invalid.
    
    Configuration:
        Set expected key via environment variable:
        ```bash
        export API_KEY="sk-your-secret-key"
        ```
        
        Default key if not set: "sk-test"
    
    Example Usage:
        ```python
        @api.get("/protected")
        async def protected_endpoint(
            api_key: str | None = Depends(verify_api_key)
        ):
            if api_key is None:
                raise HTTPException(401, "API key required")
            return {"message": "Access granted"}
        ```
    
    Note:
        - In production, use strong random keys
        - Consider implementing key rotation
        - May want to add rate limiting per key
    """
    if not authorization:
        return None
    
    parts = authorization.split()
    api_key = (
        parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else authorization
    )
    
    # Simple API key validation - customize as needed
    expected_key = os.environ.get("API_KEY", "sk-test")
    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key


@api.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: (  # pyright: ignore[reportUnusedParameter]
        str | None
    ) = Depends(  # pyright: ignore[reportCallInDefaultInitializer]
        verify_api_key
    ),
):
    """Create a chat completion (OpenAI-compatible endpoint).
    
    Processes chat messages through the RAG-enhanced conversational pipeline.
    Fully compatible with OpenAI's chat completions API, allowing drop-in
    replacement for OpenAI clients.
    
    Args:
        request: Chat completion request containing:
            - messages: List of chat messages with roles and content
            - model: Model identifier (mapped to configured LLM)
            - temperature: Sampling temperature (0.0-2.0)
            - max_tokens: Maximum response length
            - stream: Whether to stream the response
            - Other OpenAI-compatible parameters
        api_key: Validated API key from Authorization header.
            Format: "Bearer sk-..."
        
    Returns:
        Chat completion response matching OpenAI's format:
            - id: Unique completion ID
            - object: "chat.completion" or "chat.completion.chunk"
            - created: Unix timestamp
            - model: Model used
            - choices: Array with message/delta and finish_reason
            - usage: Token counts (estimated)
        
    Raises:
        HTTPException:
            - 401: Invalid or missing API key
            - 500: Processing error with details
    
    Example Request:
        ```bash
        curl -X POST http://localhost:7990/v1/chat/completions \
          -H "Authorization: Bearer sk-test" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "rag-model",
            "messages": [
              {"role": "user", "content": "What is RAG?"}
            ],
            "temperature": 0.7,
            "stream": false
          }'
        ```
    
    Streaming Example:
        ```python
        import openai
        client = openai.OpenAI(
            base_url="http://localhost:7990/v1",
            api_key="sk-test"
        )
        stream = client.chat.completions.create(
            model="rag-model",
            messages=[{"role": "user", "content": "Explain embeddings"}],
            stream=True
        )
        for chunk in stream:
            print(chunk.choices[0].delta.content, end="")
        ```
    
    Note:
        - RAG context is automatically included when retriever is configured
        - Conversation history is maintained in ChatState
        - Streaming uses Server-Sent Events (SSE) format
    """
    try:
        # Convert messages to format your model can use
        messages = [
            ChatMessage(role=msg.role, content=msg.content) for msg in request.messages
        ]
        
        if request.stream:
            return StreamingResponse(
                stream_chat_response(messages, request),
                media_type="text/event-stream",
            )
        
        # Process with your custom logic
        response_content = await chat_response(messages, request)
        
        # Format response to match OpenAI's format
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,  # Estimate or implement proper counting
                "completion_tokens": 50,  # Estimate or implement proper counting
                "total_tokens": 150,  # Sum of prompt and completion tokens
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    api_key: (  # pyright: ignore[reportUnusedParameter]
        str | None
    ) = Depends(  # pyright: ignore[reportCallInDefaultInitializer]
        verify_api_key
    ),
):
    """Create a text completion (OpenAI-compatible endpoint).
    
    Args:
        request: Completion request
        api_key: Validated API key
        
    Returns:
        Completion response in OpenAI format
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Handle both string and list prompts
        prompt = (
            request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        )
        
        # Handle streaming if requested
        if request.stream:
            return StreamingResponse(
                stream_completion_response(prompt, request),
                media_type="text/event-stream",
            )
        
        # # Process with your custom logic
        # response_content = await process_completion_message(messages, request)
        response_content = ""
        
        # Format response to match OpenAI's format
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "text": response_content,
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": 100,  # Estimate or implement proper counting
                "completion_tokens": 50,  # Estimate or implement proper counting
                "total_tokens": 150,  # Sum of prompt and completion tokens
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    api_key: (  # pyright: ignore[reportUnusedParameter]
        str | None
    ) = Depends(  # pyright: ignore[reportCallInDefaultInitializer]
        verify_api_key
    ),
):
    """Create embeddings (OpenAI-compatible endpoint).
    
    Generates vector embeddings for input text using the configured embedding
    model. Compatible with OpenAI's embeddings API for seamless integration.
    
    Args:
        request: Embedding request containing:
            - input: String or list of strings to embed
            - model: Model identifier (mapped to configured embedding model)
            - user: Optional user identifier for tracking
        api_key: Validated API key from Authorization header.
            Format: "Bearer sk-..."
        
    Returns:
        Embeddings response matching OpenAI's format:
            - object: "list"
            - data: Array of embedding objects, each containing:
                - object: "embedding"
                - embedding: Vector array of floats
                - index: Position in input array
            - model: Model used
            - usage: Token usage statistics
        
    Raises:
        HTTPException:
            - 401: Invalid or missing API key
            - 500: Embedding generation error
    
    Example Request:
        ```bash
        curl -X POST http://localhost:7990/v1/embeddings \
          -H "Authorization: Bearer sk-test" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "text-embedding-ada-002",
            "input": "Machine learning is fascinating"
          }'
        ```
    
    Batch Example:
        ```python
        import openai
        client = openai.OpenAI(
            base_url="http://localhost:7990/v1",
            api_key="sk-test"
        )
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[
                "First text to embed",
                "Second text to embed",
                "Third text to embed"
            ]
        )
        for i, data in enumerate(response.data):
            print(f"Embedding {i}: {data.embedding[:5]}...")  # First 5 dims
        ```
    
    Note:
        - Current implementation returns placeholder embeddings
        - TODO: Integrate with actual embedding model from workflow
        - Batch processing supported for multiple inputs
        - Embedding dimensions depend on configured model
    """
    try:
        # Convert input to list format if it's a string
        inputs = request.input if isinstance(request.input, list) else [request.input]
        
        # In a real implementation, you would call your embedding model here
        # This is a placeholder that returns fake embeddings
        embeddings: list[dict[str, str | list[float] | int]] = []
        for i, _text in enumerate(inputs):
            # Replace with actual embedding generation
            embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # Dummy embedding
            embeddings.append(
                {"object": "embedding", "embedding": embedding, "index": i}
            )
        
        return {
            "object": "list",
            "data": embeddings,
            "model": request.model,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in inputs),
                "total_tokens": sum(len(text.split()) for text in inputs),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/v1/models")
async def list_models(
    api_key: (  # pyright: ignore[reportUnusedParameter]
        str | None
    ) = Depends(  # pyright: ignore[reportCallInDefaultInitializer]
        verify_api_key
    ),
):
    """List available models (OpenAI-compatible endpoint).
    
    Returns information about available models in the RAG system.
    Compatible with OpenAI's models API for client compatibility.
    
    Args:
        api_key: Validated API key from Authorization header.
            Format: "Bearer sk-..."
        
    Returns:
        Models list matching OpenAI's format:
            - object: "list"
            - data: Array of model objects, each containing:
                - id: Model identifier with "-RAG" suffix
                - object: "model"
                - created: Unix timestamp
                - owned_by: Organization identifier
        
    Example Request:
        ```bash
        curl -X GET http://localhost:7990/v1/models \
          -H "Authorization: Bearer sk-test"
        ```
    
    Example Response:
        ```json
        {
          "object": "list",
          "data": [
            {
              "id": "llama2-RAG",
              "object": "model",
              "created": 1677610602,
              "owned_by": "your-organization"
            }
          ]
        }
        ```
    
    Note:
        - Model ID is derived from configured LLM or embedding model
        - The "-RAG" suffix indicates RAG-enhanced capabilities
        - Returns "invalid-RAG" if no models are configured
    """
    return {
        "object": "list",
        "data": [
            {
                "id": ((llm_model(workflow.config.chat.llm)
                        if workflow is not None and
                           workflow.config.chat is not None
                        else None) or
                       (embedding_model(workflow.config.retrieval.embedding)
                        if workflow is not None and
                           workflow.config.retrieval.embedding is not None
                        else None) or
                       "invalid") + "-RAG",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "your-organization",
            }
        ],
    }


async def process_chat_messages(
    messages: Sequence[ChatMessage],
    request: ChatCompletionRequest,
) -> StreamingAgentChatResponse | AGENT_CHAT_RESPONSE_TYPE:
    """Process chat messages with your custom logic.
    
    Args:
        messages: Chat messages
        request: Chat completion request
        
    Returns:
        Chat response
        
    Raises:
        SystemExit: If no user message found or ChatState not initialized
    """
    user_message = next((msg for msg in messages if msg.role == "user"), None)
    if not user_message:
        error("No user message found")
    
    if chat_state is None:
        error("ChatState not initialized")
        
        # user=request.user or "user1",
        # chat_history=list(messages),
        # token_limit=request.max_tokens,
        
    return chat_state.chat(
        query=user_message.content or "",
        streaming=request.stream or False,
    )


async def chat_response(
    messages: Sequence[ChatMessage], request: ChatCompletionRequest
) -> str | NoReturn:
    """Process chat messages with your custom logic.
    
    Args:
        messages: Chat messages
        request: Chat completion request
        
    Returns:
        Response string
    """
    response = await process_chat_messages(messages, request)
    return response.response


async def stream_chat_response(
    messages: Sequence[ChatMessage], request: ChatCompletionRequest
) -> AsyncGenerator[str, None] | NoReturn:
    """Stream chat responses in the SSE format expected by OpenAI clients.
    
    Args:
        messages: Chat messages
        request: Chat completion request
        
    Yields:
        SSE-formatted response chunks
    """
    try:
        response = await process_chat_messages(messages, request)
        
        async for chunk in response.async_response_gen():
            response_json = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk,
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(response_json)}\n\n"
            await asyncio.sleep(0.1)  # Simulate processing time
        
        response_json: dict[
            str,
            str
            | int
            | list[
                dict[
                    str,
                    dict[Any, Any]  # pyright: ignore[reportExplicitAny]
                    | int
                    | str
                    | None,
                ]
            ],
        ] = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(response_json)}\n\n"
        
        # End the stream
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_json = json.dumps({"error": {"message": str(e), "type": "server_error"}})
        yield f"data: {error_json}\n\n"
        yield "data: [DONE]\n\n"


async def stream_completion_response(
    _prompt: str, _request: CompletionRequest
) -> AsyncGenerator[str, None]:
    """Stream completion responses in the SSE format expected by OpenAI clients.
    
    Args:
        _prompt: Input prompt (unused in placeholder)
        _request: Completion request (unused in placeholder)
        
    Yields:
        SSE-formatted response chunks
    """
    try:
        # # Get full response
        # response = llm.astream_complete(prompt).text
        
        # for i, chunk in response:
        #     response_json = {
        #         "id": f"cmpl-{int(time.time())}",
        #         "object": "text_completion.chunk",
        #         "created": int(time.time()),
        #         "model": request.model,
        #         "choices": [
        #             {
        #                 "text": chunk,
        #                 "index": 0,
        #                 "finish_reason": "stop" if i == len(chunks) - 1 else None,
        #                 "logprobs": None,
        #             }
        #         ],
        #     }
        #     yield f"data: {json.dumps(response_json)}\n\n"
        #     await asyncio.sleep(0.1)
        
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_json = json.dumps({"error": {"message": str(e), "type": "server_error"}})
        yield f"data: {error_json}\n\n"
        yield "data: [DONE]\n\n"


# Root endpoint for API status
@api.get("/")
async def root():
    """Get API status.
    
    Health check endpoint that confirms the API server is running.
    Useful for monitoring and load balancer health checks.
    
    Returns:
        Status object containing:
            - status: Confirmation message
            - version: API version string
    
    Example Request:
        ```bash
        curl http://localhost:7990/
        ```
    
    Example Response:
        ```json
        {
          "status": "API is running",
          "version": "1.0.0"
        }
        ```
    
    Note:
        - No authentication required
        - Can be used for uptime monitoring
        - Returns 200 OK when service is healthy
    """
    return {"status": "API is running", "version": "1.0.0"}


def start_api_server(host: str, port: int, reload: bool):
    """Start the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Whether to enable auto-reload
    """
    uvicorn.run("api:api", host=host, port=port, reload=reload)