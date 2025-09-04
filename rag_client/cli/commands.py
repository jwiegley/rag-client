"""CLI command implementations for RAG client."""

import atexit
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, NoReturn, Optional

import readline
from llama_index.core.base.response.schema import Response, StreamingResponse
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryType
from llama_index.core.storage.chat_store import SimpleChatStore
from xdg_base_dirs import xdg_config_home

from ..core.models import ChatState, QueryState
from ..core.workflow import RAGWorkflow
from ..exceptions import ConfigurationError, RAGClientError, RetrievalError
from ..utils.helpers import clean_special_tokens, error
from ..utils.logging import get_logger

# Module logger
logger = get_logger(__name__)


def rag_initialize(
    logger: logging.Logger,
    config_path: Path,
    input_from: Optional[str],
    num_workers: Optional[int] = None,
    recursive: bool = False,
    index_files: bool = False,
    top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
    verbose: bool = False,
) -> tuple[RAGWorkflow, Optional[BaseRetriever]]:
    """Initialize RAG workflow and retriever.
    
    Sets up the complete RAG pipeline by loading configuration, processing input
    files if provided, and creating appropriate retriever instances. This is the
    main initialization function called by all CLI commands.
    
    Args:
        logger: Logger instance for operation logging and debugging.
        config_path: Path to YAML configuration file containing model settings,
            embedding configurations, and storage parameters.
        input_from: Optional path to input file or directory. Can be:
            - Single file: "/path/to/document.pdf"
            - Directory: "/path/to/docs/"
            - None: Load from existing storage (e.g., PostgreSQL)
        num_workers: Number of parallel workers for document processing.
            None uses sequential processing. Recommended: 4-8 for large datasets.
        recursive: If True, recursively process subdirectories when input_from
            is a directory. Default False for explicit control.
        index_files: If True, force re-indexing even if cache exists.
            Use when documents changed but cache key hasn't.
        top_k: Number of top results for dense/vector retrieval.
            Overrides config value. Typical range: 3-20.
        sparse_top_k: Number of top results for sparse/keyword retrieval.
            Only used with hybrid search configurations.
        verbose: If True, show progress bars and detailed logging.
            Helpful for debugging and monitoring long operations.
        
    Returns:
        Tuple of (RAGWorkflow instance, Optional[BaseRetriever]):
            - RAGWorkflow: Initialized workflow object for further operations
            - BaseRetriever: Configured retriever, or None if no embedding configured
    
    Raises:
        ConfigurationError: If config file is invalid or missing required fields.
        SystemExit: If critical initialization fails.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> workflow, retriever = rag_initialize(
        ...     logger=logger,
        ...     config_path=Path("config.yaml"),
        ...     input_from="/data/docs",
        ...     num_workers=4,
        ...     verbose=True
        ... )
        >>> # Now ready for search/query operations
    
    Note:
        - Configuration is cached based on input file fingerprints
        - PostgreSQL storage requires proper database setup
        - Large document sets benefit from num_workers > 1
    """
    from ..utils.helpers import read_files
    
    config = RAGWorkflow.load_config(config_path)
    
    if input_from is not None:
        input_files = read_files(input_from, recursive)
        count = str(len(input_files)) if input_files else "no"
        logger.info(f"{count} input file(s)")
    else:
        input_files = None
        logger.info("No input files")
    
    rag = RAGWorkflow(logger, config)
    
    if config.retrieval.embedding is not None:
        # If the input_files is None, the retriever might still load indices
        # from cache or a database
        retriever = rag.load_retriever(
            num_workers=num_workers,
            input_files=input_files,
            embed_individually=getattr(config.retrieval, 'embed_individually', False),
            index_files=index_files,
            top_k=top_k,
            sparse_top_k=sparse_top_k,
            verbose=verbose,
        )
    else:
        logger.info("No retriever used")
        retriever = None
    
    return (rag, retriever)


def search_command(rag: RAGWorkflow, retriever: BaseRetriever, query: str):
    """Execute a search command.
    
    Performs semantic search across indexed documents and returns relevant
    document chunks in JSON format. This is a low-level search that returns
    raw retrieval results without LLM processing.
    
    Args:
        rag: The RAG workflow object containing configuration and methods.
        retriever: The retriever object configured for document search.
            Must be initialized via rag_initialize() first.
        query: The search query string. Can be:
            - Natural language question: "What is machine learning?"
            - Keywords: "neural networks training"
            - Specific facts: "transformer architecture attention mechanism"
    
    Returns:
        None (prints JSON results to stdout).
    
    Raises:
        RetrievalError: If search execution fails with details about
            the query and underlying cause.
    
    Output Format:
        JSON array of retrieved nodes, each containing:
        - "text": The content of the document chunk
        - "metadata": Source file, page numbers, and other metadata
    
    Example:
        >>> search_command(rag, retriever, "How does RAG work?")
        [
          {
            "text": "RAG combines retrieval with generation...",
            "metadata": {
              "file_name": "rag_guide.pdf",
              "page": 5
            }
          }
        ]
    
    Note:
        - Number of results depends on top_k configuration
        - Results are ranked by relevance score
        - Use query_command for LLM-processed answers
    """
    try:
        logger.info(f"Executing search for query: {query[:100]}...")
        # Retrieve nodes using the retriever and query
        nodes = rag.retrieve_nodes(retriever, query)
        logger.debug(f"Retrieved {len(nodes) if isinstance(nodes, list) else 'unknown number of'} nodes")
        # Print the retrieved nodes in JSON format
        print(json.dumps(nodes, indent=2))
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise RetrievalError(
            message="Failed to execute search",
            query=query,
            cause=e
        )


def query_command(query_state: QueryState, query: QueryType):
    """Execute a query command.
    
    Processes a query through the full RAG pipeline: retrieval, context
    assembly, and LLM generation. Returns a complete answer synthesized
    from retrieved documents.
    
    Args:
        query_state: The query state object containing:
            - Query engine configuration
            - LLM instance for generation
            - Retriever for document search
            - Response synthesis settings
        query: The query string to process. Supports:
            - Questions: "Explain the architecture of transformers"
            - Commands: "Summarize the key points about..."
            - Analysis: "Compare X and Y approaches"
    
    Returns:
        None (prints response to stdout).
    
    Raises:
        RAGClientError: If query execution fails, with context about
            the query and detailed error information.
    
    Response Types:
        - StreamingResponse: Tokens printed as generated (if streaming enabled)
        - Response: Complete response printed at once
    
    Example:
        >>> # Non-streaming query
        >>> query_command(query_state, "What are the main components of RAG?")
        RAG consists of three main components: a retriever for finding
        relevant documents, a context processor for organizing information,
        and a generator for producing answers...
        
        >>> # Streaming query (with streaming=True in config)
        >>> query_command(query_state, "Explain embeddings")
        Embeddings are dense vector representations... [streamed output]
    
    Note:
        - Response quality depends on retriever and LLM configuration
        - Streaming provides faster initial response but same total time
        - Context window limits may truncate very long contexts
    """
    try:
        logger.info(f"Executing query: {query[:100]}...")
        # Execute the query using the query state
        response = query_state.query(query=query)
        logger.debug(f"Query response type: {type(response).__name__}")
        
        # Handle different response types
        match response:
            case StreamingResponse():
                logger.debug("Processing streaming response")
                # Print the response tokens as they are generated
                token_count = 0
                for token in response.response_gen:
                    token = clean_special_tokens(token)
                    print(token, end="", flush=True)
                    token_count += 1
                print()
                logger.debug(f"Streamed {token_count} tokens")
            case Response():
                # Print the response
                logger.debug(f"Response length: {len(response.response)} characters")
                print(response.response)
            case _:
                # Log an error for unhandled response types
                logger.error(f"query_command cannot render response type: {type(response)}")
                raise ValueError(f"Unhandled response type: {type(response)}")
                
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise RAGClientError(
            message="Failed to execute query",
            context={"query": query[:100]},
            cause=e
        )


def chat_command(
    cmd_logger: logging.Logger,  # Renamed to avoid confusion with module logger
    chat_state: ChatState,
    query: str,
    streaming: bool,
):
    """Execute a chat command.
    
    Processes a chat message through the conversational AI pipeline with
    context awareness and history management. Supports both streaming and
    non-streaming responses.
    
    Args:
        cmd_logger: Command-specific logger for operation tracking.
            Separate from module logger to avoid naming conflicts.
        chat_state: The chat state object containing:
            - Chat engine configuration
            - Conversation history
            - Memory buffer management
            - Optional retriever for RAG-enhanced chat
        query: The user's message or question. Can be:
            - Follow-up questions referencing previous context
            - New topics (context automatically managed)
            - Commands like "summarize our discussion"
        streaming: If True, stream response tokens as generated.
            If False, return complete response at once.
    
    Returns:
        None (prints response to stdout).
    
    Raises:
        RAGClientError: If chat processing fails, with context about
            the query, streaming mode, and underlying error.
    
    Streaming Behavior:
        - True: Tokens appear immediately as generated (better UX)
        - False: Complete response appears at once (better for logging)
    
    Example:
        >>> # Non-streaming chat
        >>> chat_command(logger, chat_state, "Hello, how are you?", False)
        I'm doing well, thank you! How can I assist you today?
        
        >>> # Streaming chat with follow-up
        >>> chat_command(logger, chat_state, "Tell me about Python", True)
        Python is a high-level programming... [streamed]
        >>> chat_command(logger, chat_state, "What are its main uses?", True)
        Based on our discussion about Python... [streamed with context]
    
    Note:
        - Chat history is maintained across calls in the same session
        - Memory buffer limits prevent context overflow
        - RAG retrieval enhances responses when configured
    """
    try:
        # Execute the chat using the chat state
        logger.info(f"Processing chat query: {query[:100]}...")
        logger.debug(f"Chat mode: {'streaming' if streaming else 'non-streaming'}")
        
        response = chat_state.chat(
            query=query,
            streaming=streaming,
        )
        
        # Handle streaming and non-streaming responses
        if streaming:
            logger.debug("Processing streaming chat response")
            token_count = 0
            # Print the response tokens as they are generated
            for token in response.response_gen:
                token = clean_special_tokens(token)
                print(token, end="", flush=True)
                token_count += 1
            print()
            logger.debug(f"Streamed {token_count} tokens in chat response")
        else:
            # Print the response
            logger.debug(f"Chat response length: {len(response.response)} characters")
            print(response.response)
            
    except Exception as e:
        logger.error(f"Chat command failed: {e}")
        raise RAGClientError(
            message="Failed to process chat",
            context={"query": query[:100], "streaming": streaming},
            cause=e
        )


def execute_command(
    logger: logging.Logger,
    rag: RAGWorkflow,
    retriever: Optional[BaseRetriever],
    args: Any,
):
    """Execute the RAG client commands.
    
    Main command dispatcher that routes CLI commands to appropriate handlers.
    Manages the execution flow, validates requirements, and handles the
    interactive chat loop when needed.
    
    Args:
        logger: Logger instance for operation tracking and debugging.
        rag: The RAG workflow object containing configuration and methods.
        retriever: Optional retriever for document search. Required for
            search command, optional for query/chat with RAG enhancement.
        args: Parsed command-line arguments containing:
            - command: The command to execute (search/query/chat)
            - args: Command-specific arguments
            - streaming: Whether to use streaming responses
            - verbose: Whether to show detailed output
    
    Command Behaviors:
        - "search": Direct retrieval, requires retriever
        - "query": One-shot Q&A, optional retriever for RAG
        - "chat": Interactive conversation with history
    
    Chat Mode Features:
        - Readline history (saved to ~/.config/rag-client/chat_history)
        - Special commands: "exit", "quit" to leave
        - Inline search: "search <query>" within chat
        - Inline query: "query <query>" within chat
        - Persistent chat history (if configured)
    
    Raises:
        ConfigurationError: If required components are not configured:
            - Search without retriever
            - Query without query engine config
            - Chat without chat engine config
        SystemExit: Via error() for unrecognized commands.
    
    Example Usage:
        >>> # Search command
        >>> args = Namespace(command="search", args=["machine learning"])
        >>> execute_command(logger, rag, retriever, args)
        
        >>> # Interactive chat
        >>> args = Namespace(command="chat", streaming=True)
        >>> execute_command(logger, rag, retriever, args)
        user> Hello!
        [Assistant responds...]
        user> exit
        Goodbye!
    
    Note:
        - Chat history persists across sessions if keep_history=True
        - Query and chat states are lazily initialized for efficiency
        - Supports hot-swapping between commands in chat mode
    """
    # Handle different commands
    match args.command:
        case "search":
            # Check if a retriever is available
            if retriever is not None:
                logger.debug("Retriever available, executing search")
                search_command(rag, retriever, args.args[0])
            else:
                logger.error("Search command requires a retriever")
                raise ConfigurationError(
                    message="Search command requires a retriever to be configured",
                    field="retriever"
                )
        case "query":
            if rag.config.query is None:
                logger.error("Query command requires query engine configuration")
                raise ConfigurationError(
                    message="'query' command requires query engine to be configured",
                    field="query"
                )
            # Create a query state object
            llm = rag.realize_llm(rag.config.query.llm, verbose=args.verbose)
            query_state = QueryState(
                config=rag.config.query,
                llm=llm,
                retriever=retriever,
                streaming=args.streaming,
                verbose=args.verbose,
            )
            # Execute the query command
            query_command(query_state, query=args.args[0])
        case "chat":
            if rag.config.chat is None:
                logger.error("Chat command requires chat engine configuration")
                raise ConfigurationError(
                    message="'chat' command requires chat engine to be configured",
                    field="chat"
                )

            # Path to the history file (change as needed)
            HISTFILE = xdg_config_home() / "rag-client" / "chat_history"

            # Load history if it exists
            try:
                readline.read_history_file(HISTFILE)
                logger.debug(f"Loaded chat history from {HISTFILE}")
            except FileNotFoundError:
                logger.debug(f"No chat history file found at {HISTFILE}")
                pass
            except Exception as e:
                logger.warning(f"Failed to load chat history: {e}")

            # Optionally limit the history length
            readline.set_history_length(1000)

            # Save history on exit
            _ = atexit.register(readline.write_history_file, HISTFILE)

            # Optional: Enable tab completion (if you want)
            # readline.parse_and_bind("tab: complete")

            user = rag.config.chat.default_user or "user"

            # Create a chat store object
            chat_store_json = xdg_config_home() / "rag-client" / "chat_store.json"
            if rag.config.chat.keep_history:
                chat_store = SimpleChatStore.from_persist_path(str(chat_store_json))
            else:
                chat_store = SimpleChatStore()

            query_state: Optional[QueryState] = None
            chat_state: Optional[ChatState] = None

            # Enter a chat loop
            while True:
                query = input(f"\n{user}> ")
                if query.strip().lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    # Persist the chat store if required
                    if rag.config.chat.keep_history:
                        chat_store.persist(persist_path=str(chat_store_json))
                    break
                elif query.startswith("search "):
                    # Handle search queries within the chat loop
                    if retriever is not None:
                        search_command(rag, retriever, query[7:])
                    else:
                        print("ERROR: Search command requires a retriever")
                elif query.startswith("query "):
                    if rag.config.query is None:
                        print("ERROR: 'query' command requires engine to be configured")
                    else:
                        # Handle query commands within the chat loop
                        if query_state is None:
                            llm = rag.realize_llm(
                                rag.config.query.llm, verbose=args.verbose
                            )
                            query_state = QueryState(
                                config=rag.config.query,
                                llm=llm,
                                retriever=retriever,
                                streaming=args.streaming,
                                verbose=args.verbose,
                            )
                        query_command(query_state, query[6:])
                else:
                    # Handle chat queries
                    if chat_state is None:
                        logger.debug("Realize chat LLM")
                        llm = rag.realize_llm(
                            rag.config.chat.llm,
                            verbose=args.verbose,
                        )
                        logger.debug("Initialize chat state")
                        chat_state = ChatState(
                            config=rag.config.chat,
                            llm=llm,
                            user="user",
                            retriever=retriever,
                            verbose=args.verbose,
                        )
                    logger.debug("Handle chat command")
                    chat_command(
                        logger,
                        chat_state,
                        query=query,
                        streaming=args.streaming,
                    )
        case _:
            # Log an error for unrecognized commands
            error(f"Command unrecognized: {args.command}")


# Public command wrapper functions for easy import
def cmd_index(*args, **kwargs) -> None:
    """Index command (no-op, indexing done during initialization)."""
    pass


def cmd_search(rag: RAGWorkflow, retriever: BaseRetriever, query: str) -> None:
    """Execute search command."""
    search_command(rag, retriever, query)


def cmd_query(query_state: QueryState, query: str) -> None:
    """Execute query command."""
    query_command(query_state, query)


def cmd_chat(
    logger: logging.Logger,
    chat_state: ChatState,
    query: str,
    streaming: bool = False,
) -> None:
    """Execute chat command."""
    chat_command(logger, chat_state, query, streaming)


def cmd_serve(host: str = "localhost", port: int = 7990, reload: bool = False) -> None:
    """Start API server.
    
    Args:
        host: Server host
        port: Server port
        reload: Whether to reload on source changes
    """
    import uvicorn
    uvicorn.run(
        "rag_client.api.server:api",
        host=host,
        port=port,
        reload=reload,
    )