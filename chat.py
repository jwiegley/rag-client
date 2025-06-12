from typing import override
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
    StreamingAgentChatResponse,
    AgentChatResponse,
)
from llama_index.core.llms import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    MessageRole,
)
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,  # pyright: ignore[reportUnknownVariableType]
)
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.tools import ToolOutput


class SimpleContextChatEngine(BaseChatEngine):
    """
    A simplified context-aware chat engine that retrieves context for each query
    but synthesizes responses using all chunks at once (up to context limit).

    Unlike ContextChatEngine, this avoids iterative refinement to improve performance
    when top_k is large.
    """

    _retriever: BaseRetriever
    _llm: LLM
    _memory: BaseMemory
    _prefix_messages: list[ChatMessage]
    _node_postprocessors: list[BaseNodePostprocessor]
    _context_window: int
    _verbose: bool
    _callback_manager: CallbackManager
    _response_mode: ResponseMode

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory | None = None,
        prefix_messages: list[ChatMessage] | None = None,
        node_postprocessors: list[BaseNodePostprocessor] | None = None,
        context_window: int = 3900,  # Leave room for response
        response_mode: ResponseMode = ResponseMode.COMPACT,
        verbose: bool = False,
        callback_manager: CallbackManager | None = None,
    ):
        """
        Initialize the SimpleContextChatEngine.

        Args:
            retriever: The retriever to fetch relevant context
            llm: Language model to use (defaults to Settings.llm)
            memory: Chat memory buffer (defaults to ChatMemoryBuffer)
            prefix_messages: Initial system/context messages
            node_postprocessors: Optional postprocessors for retrieved nodes
            response_mode: Response synthesis mode (default: COMPACT for efficiency)
            verbose: Whether to print verbose output
            callback_manager: Callback manager for tracking events
        """
        self._retriever = retriever
        self._llm = llm or Settings.llm
        self._memory = (
            memory
            or ChatMemoryBuffer.from_defaults(  # pyright: ignore[reportUnknownMemberType]
                llm=self._llm,
            )
        )
        self._prefix_messages = prefix_messages or []
        self._node_postprocessors = node_postprocessors or []
        self._context_window = context_window
        self._verbose = verbose
        self._callback_manager = callback_manager or CallbackManager()
        self._response_mode = response_mode

    def _get_response_synthesizer(
        self, _chat_history: list[ChatMessage], streaming: bool = False
    ) -> BaseSynthesizer:
        return get_response_synthesizer(
            llm=self._llm,
            response_mode=self._response_mode,
            streaming=streaming,
            verbose=self._verbose,
            callback_manager=self._callback_manager,
        )

    @property
    @override
    def chat_history(self) -> list[ChatMessage]:
        """Get the chat history."""
        return self._memory.get_all()

    @override
    def reset(self) -> None:
        """Reset the chat engine state (clears memory)."""
        self._memory.reset()

    def _retrieve_context(self, message: str) -> list[NodeWithScore]:
        """
        Retrieve relevant context for the given message.

        Args:
            message: The user's query message

        Returns:
            list of retrieved nodes with scores
        """
        query_bundle = QueryBundle(query_str=message)
        nodes: list[NodeWithScore] = self._retriever.retrieve(query_bundle)

        # Apply any postprocessors
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes,
                query_bundle=query_bundle,
            )

        return nodes

    def _build_context_prompt(
        self,
        message: str,
        nodes: list[NodeWithScore],
    ) -> str:
        """
        Build a prompt that includes the retrieved context.
        """
        # Extract text content from nodes
        context_strs: list[str] = []
        total_length = 0

        for node in nodes:
            if hasattr(node.node, "get_content"):
                content = node.node.get_content()
            else:
                content = str(node.node)

            # Rough token estimation (4 chars ≈ 1 token)
            estimated_tokens = len(content) // 4

            if total_length + estimated_tokens > self._context_window:
                if self._verbose:
                    print(
                        f"Truncating context at {len(context_strs)} chunks due to context window"
                    )
                break

            context_strs.append(content)
            total_length += estimated_tokens

        # Combine contexts
        full_context = "\n\n".join(context_strs)

        # Format prompt with context
        context_prompt = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{full_context}\n"
            f"---------------------\n"
            f"Given the context information and chat history, "
            f"answer the query.\n"
            f"Query: {message}\n"
            f"Answer: "
        )

        return context_prompt

    def _prepare_chat_messages(
        self, message: str, nodes: list[NodeWithScore]
    ) -> list[ChatMessage]:
        """
        Prepare the full list of messages including context for the LLM.
        """
        messages: list[ChatMessage] = []

        # Add prefix messages (system prompts, etc.)
        messages.extend(self._prefix_messages)

        # Add chat history
        messages.extend(self._memory.get_all())

        # Create context-enhanced user message
        context_prompt = self._build_context_prompt(message, nodes)
        messages.append(ChatMessage(role=MessageRole.USER, content=context_prompt))

        return messages

    @override
    def chat(
        self,
        message: str,
        chat_history: list[ChatMessage] | None = None,
    ) -> AgentChatResponse:
        """
        Main chat interface - retrieves context and generates response.
        """
        # Retrieve relevant context
        if self._verbose:
            print(f"Retrieving context for: {message}")

        nodes = self._retrieve_context(message)

        if self._verbose:
            print(f"Retrieved {len(nodes)} nodes")

        # Override chat history if provided
        if chat_history is not None:
            self._memory.set(chat_history)

        # Prepare messages with context
        messages = self._prepare_chat_messages(message, nodes)

        # Get response from LLM
        response: ChatResponse = self._llm.chat(messages)

        # Update memory with the original message (not the context-enhanced one)
        self._memory.put(
            ChatMessage(
                role=MessageRole.USER,
                content=message,
            )
        )
        self._memory.put(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.message.content,
            )
        )

        # Return formatted response
        return AgentChatResponse(
            response=response.message.content or "",
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=nodes,
                )
            ],
            source_nodes=nodes,
        )

    @override
    def stream_chat(
        self,
        message: str,
        chat_history: list[ChatMessage] | None = None,
    ) -> StreamingAgentChatResponse:
        """
        Streaming version of chat - retrieves context and streams response.
        """
        # Retrieve relevant context
        if self._verbose:
            print(f"Retrieving context for: {message}")

        nodes = self._retrieve_context(message)

        if self._verbose:
            print(f"Retrieved {len(nodes)} nodes")

        # Override chat history if provided
        if chat_history is not None:
            self._memory.set(chat_history)

        # Prepare messages with context
        messages = self._prepare_chat_messages(message, nodes)

        # Get streaming response from LLM
        response_stream = self._llm.stream_chat(messages)

        # Update memory with the original message
        self._memory.put(ChatMessage(role=MessageRole.USER, content=message))

        # We need to accumulate the response for memory
        accumulated_response: list[str] = []

        def response_generator() -> ChatResponseGen:
            for chunk in response_stream:
                if chunk.delta:
                    accumulated_response.append(chunk.delta)
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT, content=chunk.delta
                        )
                    )

            # After streaming is complete, update memory
            full_response = "".join(accumulated_response)
            self._memory.put(
                ChatMessage(role=MessageRole.ASSISTANT, content=full_response)
            )

        # Return streaming response
        return StreamingAgentChatResponse(
            chat_stream=response_generator(),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=nodes,
                )
            ],
            source_nodes=nodes,
        )

    @override
    async def achat(
        self,
        message: str,
        chat_history: list[ChatMessage] | None = None,
    ) -> AgentChatResponse:
        """
        Async version of chat method.

        Args:
            message: User's input message
            chat_history: Optional override for chat history

        Returns:
            AgentChatResponse containing the assistant's response
        """
        # For now, using sync version - you can implement full async later
        return self.chat(message, chat_history)

    @override
    async def astream_chat(
        self,
        message: str,
        chat_history: list[ChatMessage] | None = None,
    ) -> StreamingAgentChatResponse:
        """
        Async streaming version of chat method.

        Args:
            message: User's input message
            chat_history: Optional override for chat history

        Returns:
            StreamingAgentChatResponse for real-time response streaming
        """
        # For now, using sync version - you can implement full async later
        return self.stream_chat(message, chat_history)
