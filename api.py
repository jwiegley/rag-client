import asyncio
import json
import os
import time
import uvicorn

from collections.abc import AsyncGenerator, Sequence
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Any, NoReturn

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
    """Process chat messages with your custom logic."""
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
    """Process chat messages with your custom logic."""
    response = await process_chat_messages(messages, request)
    return response.response


async def stream_chat_response(
    messages: Sequence[ChatMessage], request: ChatCompletionRequest
) -> AsyncGenerator[str, None] | NoReturn:
    """Stream chat responses in the SSE format expected by OpenAI clients."""
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
    """Stream completion responses in the SSE format expected by OpenAI clients."""
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
    return {"status": "API is running", "version": "1.0.0"}


def start_api_server(host: str, port: int, reload: bool):
    uvicorn.run("api:api", host=host, port=port, reload=reload)
