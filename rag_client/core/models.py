"""Core data models for RAG client."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, NoReturn, Optional, TypeAlias, override

from dataclass_wizard import JSONFileWizard, JSONWizard
from llama_index.core import PromptTemplate
from llama_index.core.chat_engine import (
    CondensePlusContextChatEngine,
    ContextChatEngine,
    SimpleChatEngine,
)
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.evaluation import (
    BaseEvaluator,
    GuidelineEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import (
    BaseQueryEngine,
    CitationQueryEngine,
    RetrieverQueryEngine,
    RetryQueryEngine,
    RetrySourceQueryEngine,
)
from llama_index.core.query_engine.custom import STR_OR_RESPONSE_TYPE, CustomQueryEngine
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, Node, QueryType
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.storage.chat_store import SimpleChatStore
from pydantic import BaseModel


def error(msg: str) -> NoReturn:
    """Print error message and exit.
    
    Args:
        msg: Error message to display
        
    Raises:
        SystemExit: Always exits with code 1
    """
    print(msg, file=sys.stderr)
    sys.exit(1)


class SimpleQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""
    
    llm: LLM
    response_synthesizer: BaseSynthesizer
    qa_prompt: PromptTemplate = field(
        default=PromptTemplate("Query: {query_str}\nAnswer: ")
    )
    
    def __init__(self, **kwargs: Any):
        self.response_synthesizer = kwargs[
            "response_synthesizer"
        ] or get_response_synthesizer(response_mode=ResponseMode.COMPACT)
        super().__init__(**kwargs)
    
    @override
    def custom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
        response = self.llm.complete(self.qa_prompt.format(query_str=query_str))
        return str(response)


FunctionsType: TypeAlias = list[dict[str, Any | None]] | None


class Message(BaseModel):
    """OpenAI-compatible message."""
    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.
    
    https://platform.openai.com/docs/api-reference/chat/create
    """
    
    messages: list[Message]
    model: str
    # audio
    frequency_penalty: float | None = 0
    function_call: str | dict[str, str | None] | None = None
    # ^ instead use tool_choice
    functions: FunctionsType = None
    # ^ instead use tools
    # logit_bias
    # logprobs
    # max_completion_tokens
    max_tokens: int | None = None
    # ^ instead use max_completion_tokens
    # metadata
    # modalities
    n: int | None = 1
    # parallel_tool_calls
    # prediction
    presence_penalty: float | None = 0
    # reasoning_effort: str | None = None
    # response_format
    # seed
    # service_tier
    # stop
    # store
    stream: bool | None = False
    # stream_options
    temperature: float | None = 0.7
    # tool_choice
    # tools
    # top_logprobs
    top_p: float | None = 1.0
    user: str | None = None
    # web_search_options


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str
    prompt: str | list[str]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    max_tokens: int | None = 16
    presence_penalty: float | None = 0
    frequency_penalty: float | None = 0
    user: str | None = None


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    model: str
    input: str | list[str]
    user: str | None = None


@dataclass
class EmbeddedNode(JSONWizard):
    """Node with embedding information."""
    ident: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
    
    @classmethod
    def from_node(cls, node: BaseNode):
        """Create from a BaseNode.
        
        Args:
            node: Source node
            
        Returns:
            EmbeddedNode instance
            
        Raises:
            SystemExit: If node has no embedding
        """
        if node.embedding is None:
            error("Cannot construct EmbeddedNode from node with no embedding")
        return cls(
            ident=node.id_,
            content=node.get_content(),
            embedding=node.embedding,
            metadata=node.metadata,
        )
    
    def to_node(self) -> Node:
        """Convert to a Node.
        
        Returns:
            Node instance
        """
        node = Node()
        node.node_id = self.ident
        node.set_content(self.content)
        node.embedding = self.embedding
        node.metadata = self.metadata
        return node


@dataclass
class EmbeddedFile(JSONFileWizard):
    """File with embedded nodes."""
    file_path: Path
    embedded_nodes: list[EmbeddedNode]


class QueryState:
    """Query state management for RAG pipeline.
    
    Manages query engines and their configuration for document Q&A.
    """
    
    query_engine: BaseQueryEngine
    
    def __init__(
        self,
        config: 'QueryConfig',
        llm: LLM,
        retriever: Optional[BaseRetriever] = None,
        streaming: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize query state.
        
        Args:
            config: Query configuration
            llm: Language model to use
            retriever: Optional retriever for document search
            streaming: Whether to enable streaming responses
            verbose: Whether to show verbose output
        """
        from ..config.models import (
            BaseQueryEngineConfig,
            CitationQueryEngineConfig,
            EvaluatorConfig,
            GuidelineConfig,
            MultiStepQueryEngineConfig,
            QueryConfig,
            QueryEngineConfig,
            RelevancyEvaluatorConfig,
            RetrieverQueryEngineConfig,
            RetryQueryEngineConfig,
            RetrySourceQueryEngineConfig,
            SimpleQueryEngineConfig,
        )
        from .workflow import RAGWorkflow
        
        self.query_engine = self.__load_query_engine(
            config.engine or SimpleQueryEngineConfig(),
            llm,
            retriever,
            streaming,
            verbose,
        )
    
    def query(self, query: QueryType) -> RESPONSE_TYPE:
        """Execute a query synchronously."""
        return self.query_engine.query(query)
    
    async def aquery(self, query: QueryType) -> RESPONSE_TYPE:
        """Execute a query asynchronously."""
        return await self.query_engine.aquery(query)
    
    @classmethod
    def __load_base_query_engine(
        cls,
        config: 'BaseQueryEngineConfig',
        llm: LLM,
        retriever: Optional[BaseRetriever] = None,
        streaming: bool = False,
        verbose: bool = False,
    ) -> BaseQueryEngine:
        """Load base query engine from configuration."""
        from ..config.models import (
            CitationQueryEngineConfig,
            RetrieverQueryEngineConfig,
            SimpleQueryEngineConfig,
        )
        
        match config:
            case CitationQueryEngineConfig():
                if retriever is None:
                    error("CitationQueryEngine requires a retriever")
                return CitationQueryEngine(
                    retriever=retriever,
                    llm=llm,
                    citation_chunk_size=config.chunk_size,
                    citation_chunk_overlap=config.chunk_overlap,
                )
            
            case RetrieverQueryEngineConfig():
                if retriever is None:
                    error("RetrieverQueryEngine requires a retriever")
                return RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    llm=llm,
                    streaming=streaming,
                    response_mode=config.response_mode,
                    verbose=verbose,
                )
            
            case SimpleQueryEngineConfig():
                return SimpleQueryEngine(llm=llm)
            
            case _:
                error(f"Unknown query engine config type: {type(config)}")
    
    @classmethod
    def __load_query_engine(
        cls,
        config: 'QueryEngineConfig',
        llm: LLM,
        retriever: Optional[BaseRetriever] = None,
        streaming: bool = False,
        verbose: bool = False,
    ) -> BaseQueryEngine:
        """Load query engine from configuration."""
        from ..config.models import (
            CitationQueryEngineConfig,
            MultiStepQueryEngineConfig,
            RetrieverQueryEngineConfig,
            RetryQueryEngineConfig,
            RetrySourceQueryEngineConfig,
            SimpleQueryEngineConfig,
        )
        from .workflow import RAGWorkflow
        
        match config:
            case (
                CitationQueryEngineConfig()
                | RetrieverQueryEngineConfig()
                | SimpleQueryEngineConfig()
            ):
                return cls.__load_base_query_engine(
                    config,
                    llm,
                    retriever,
                    streaming,
                    verbose,
                )
            
            case MultiStepQueryEngineConfig():
                error("MultiStepQueryEngineConfig not implemented yet")
            
            case RetrySourceQueryEngineConfig():
                query_engine = cls.__load_base_query_engine(
                    config.engine, llm, retriever, streaming, verbose
                )
                if isinstance(query_engine, RetrieverQueryEngine):
                    evaluator = cls.__load_evaluator(config.evaluator, verbose)
                    return RetrySourceQueryEngine(
                        query_engine,
                        evaluator=evaluator,
                        llm=RAGWorkflow.realize_llm(config.llm, verbose),
                    )
                else:
                    error(
                        "Base engine for RetrySourceQueryEngine must be RetrieverQueryEngine"
                    )
            
            case RetryQueryEngineConfig():
                query_engine = cls.__load_base_query_engine(
                    config.engine, llm, retriever, streaming, verbose
                )
                evaluator = cls.__load_evaluator(config.evaluator, verbose)
                return RetryQueryEngine(
                    query_engine,
                    evaluator=evaluator,
                )
            
            case _:
                error(f"Unknown query engine config type: {type(config)}")
    
    @classmethod
    def __load_evaluator(
        cls,
        config: 'EvaluatorConfig',
        verbose: bool = False,
    ) -> BaseEvaluator:
        """Load evaluator from configuration."""
        from ..config.models import (
            GuidelineConfig,
            RelevancyEvaluatorConfig,
        )
        from .workflow import RAGWorkflow
        
        match config:
            case RelevancyEvaluatorConfig():
                return RelevancyEvaluator(
                    llm=RAGWorkflow.realize_llm(config.llm, verbose),
                )
            case GuidelineConfig():
                return GuidelineEvaluator(
                    llm=RAGWorkflow.realize_llm(config.llm, verbose),
                    guidelines=config.guidelines,
                )
            case _:
                error(f"Unknown evaluator config type: {type(config)}")


class ChatState:
    """Chat state management for conversational AI.
    
    Manages chat engines, history, and context for conversational interactions.
    """
    
    chat_engine: BaseChatEngine
    
    def __init__(
        self,
        config: 'ChatConfig',
        llm: LLM,
        user: str,
        chat_store: Optional[SimpleChatStore] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        retriever: Optional[BaseRetriever] = None,
        token_limit: int = 1500,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize chat state.
        
        Args:
            config: Chat configuration
            llm: Language model to use
            user: User identifier
            chat_store: Optional chat store for persistence
            chat_history: Optional initial chat history
            retriever: Optional retriever for RAG
            token_limit: Maximum tokens in context
            system_prompt: Optional system prompt
            verbose: Whether to show verbose output
        """
        from ..config.models import (
            ChatConfig,
            ChatEngineConfig,
            CondensePlusContextChatEngineConfig,
            ContextChatEngineConfig,
            SimpleChatEngineConfig,
            SimpleContextChatEngineConfig,
        )
        
        chat_store = chat_store or SimpleChatStore()
        if chat_history is not None:
            chat_store.set_messages(key=user, messages=chat_history)
        
        memory = ChatMemoryBuffer.from_defaults(
            chat_store=chat_store,
            chat_store_key=user,
            token_limit=token_limit,
        )
        
        self.chat_engine = self.__load_chat_engine(
            config.engine or SimpleChatEngineConfig(),
            llm,
            memory,
            retriever,
            config.buffer,
            config.summary_buffer,
            system_prompt,
            verbose,
        )
    
    @classmethod
    def __load_chat_engine(
        cls,
        config: 'ChatEngineConfig',
        llm: LLM,
        memory: ChatMemoryBuffer,
        retriever: Optional[BaseRetriever] = None,
        buffer: int = 10,
        summary_buffer: Optional[int] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> BaseChatEngine:
        """Load chat engine from configuration."""
        from ..config.models import (
            CondensePlusContextChatEngineConfig,
            ContextChatEngineConfig,
            SimpleChatEngineConfig,
            SimpleContextChatEngineConfig,
        )
        
        match config:
            case SimpleChatEngineConfig():
                return SimpleChatEngine.from_defaults(
                    llm=llm,
                    memory=memory,
                    system_prompt=system_prompt,
                    verbose=verbose,
                )
            
            case SimpleContextChatEngineConfig():
                if retriever is None:
                    error("Context chat engines require a retriever")
                return ContextChatEngine.from_defaults(
                    retriever=retriever,
                    llm=llm,
                    memory=memory,
                    system_prompt=system_prompt,
                    verbose=verbose,
                    context_window=config.context_window,
                )
            
            case ContextChatEngineConfig():
                if retriever is None:
                    error("Context chat engines require a retriever")
                return ContextChatEngine.from_defaults(
                    retriever=retriever,
                    llm=llm,
                    memory=memory,
                    system_prompt=system_prompt,
                    verbose=verbose,
                )
            
            case CondensePlusContextChatEngineConfig():
                if retriever is None:
                    error("Context chat engines require a retriever")
                return CondensePlusContextChatEngine.from_defaults(
                    retriever=retriever,
                    llm=llm,
                    memory=memory,
                    system_prompt=system_prompt,
                    verbose=verbose,
                    skip_condense=config.skip_condense,
                )
            
            case _:
                error(f"Unknown chat engine config type: {type(config)}")