"""MCP tool definitions for RAG operations.

This module defines the tool schemas and implementations for MCP-compatible
AI agent integration.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.models import Config
from ..core.models import ChatState, QueryState
from ..core.workflow import RAGWorkflow
from ..exceptions import ConfigurationError, RAGClientError
from ..schemas import (
    IndexingResult,
    IndexStatus,
    QueryResponse,
    SearchResponse,
    SearchResult,
    search_result_from_node,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolDefinition:
    """MCP tool definition schema."""

    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class RAGTools:
    """RAG tool implementations for MCP server.

    Provides tool definitions and execution for:
    - search: Semantic search across indexed documents
    - query: AI-generated answers from document context
    - index: Index new documents into the collection
    - get_status: Get current index status and capabilities

    Example:
        >>> tools = RAGTools(workflow, retriever)
        >>> result = await tools.execute("search", {"query": "machine learning"})
    """

    def __init__(
        self,
        workflow: RAGWorkflow,
        retriever: Optional[Any] = None,
        config: Optional[Config] = None,
    ):
        """Initialize RAG tools.

        Args:
            workflow: RAGWorkflow instance
            retriever: Optional pre-loaded retriever
            config: Optional configuration override
        """
        self.workflow = workflow
        self.retriever = retriever
        self.config = config or workflow.config
        self._query_state: Optional[QueryState] = None
        self._chat_state: Optional[ChatState] = None

    @property
    def tool_definitions(self) -> List[ToolDefinition]:
        """Get all available tool definitions."""
        return [
            self._search_tool(),
            self._query_tool(),
            self._index_tool(),
            self._status_tool(),
        ]

    def _search_tool(self) -> ToolDefinition:
        return ToolDefinition(
            name="rag_search",
            description=(
                "Search indexed documents using semantic similarity. "
                "Returns relevant document chunks with metadata. "
                "Use this when you need to find specific information in the knowledge base."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - can be a question, keywords, or natural language",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )

    def _query_tool(self) -> ToolDefinition:
        return ToolDefinition(
            name="rag_query",
            description=(
                "Ask a question and get an AI-generated answer based on indexed documents. "
                "The answer includes source citations. "
                "Use this when you need a synthesized answer rather than raw document chunks."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question to answer using the knowledge base",
                    },
                    "include_sources": {
                        "type": "boolean",
                        "description": "Whether to include source documents (default: true)",
                        "default": True,
                    },
                },
                "required": ["question"],
            },
        )

    def _index_tool(self) -> ToolDefinition:
        return ToolDefinition(
            name="rag_index",
            description=(
                "Index new documents into the knowledge base. "
                "Supports files and directories. "
                "Use this to add new information to the searchable collection."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File or directory paths to index",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Recursively index directories (default: true)",
                        "default": True,
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force re-indexing even if cached (default: false)",
                        "default": False,
                    },
                },
                "required": ["paths"],
            },
        )

    def _status_tool(self) -> ToolDefinition:
        return ToolDefinition(
            name="rag_status",
            description=(
                "Get the current status of the RAG system. "
                "Returns information about indexed documents, models, and capabilities."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

    async def execute(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool name is unknown
        """
        start_time = time.time()

        try:
            if tool_name == "rag_search":
                result = await self._execute_search(arguments)
            elif tool_name == "rag_query":
                result = await self._execute_query(arguments)
            elif tool_name == "rag_index":
                result = await self._execute_index(arguments)
            elif tool_name == "rag_status":
                result = await self._execute_status(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            return {
                "success": True,
                "result": result,
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

    async def _execute_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search tool."""
        query = args["query"]
        top_k = args.get("top_k", 5)

        if self.retriever is None:
            raise ConfigurationError("No retriever available - index documents first")

        start_time = time.time()
        nodes = self.retriever.retrieve(query)

        results = []
        for i, node in enumerate(nodes[:top_k]):
            result = search_result_from_node(node)
            results.append(result.model_dump())

        return SearchResponse(
            query=query,
            results=[SearchResult(**r) for r in results],
            total_results=len(results),
            processing_time_ms=(time.time() - start_time) * 1000,
        ).model_dump()

    async def _execute_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query tool."""
        question = args["question"]
        include_sources = args.get("include_sources", True)

        if self.config.query is None:
            raise ConfigurationError("Query engine not configured")

        if self._query_state is None:
            llm = self.workflow.realize_llm(self.config.query.llm)
            self._query_state = QueryState(
                config=self.config.query,
                llm=llm,
                retriever=self.retriever,
                streaming=False,
                verbose=False,
            )

        start_time = time.time()
        response = self._query_state.query(question)

        sources = []
        if include_sources and hasattr(response, "source_nodes"):
            for node in response.source_nodes[:5]:
                sources.append(search_result_from_node(node).model_dump())

        from ..config.models import llm_model

        return QueryResponse(
            query=question,
            answer=response.response
            if hasattr(response, "response")
            else str(response),
            sources=[SearchResult(**s) for s in sources],
            processing_time_ms=(time.time() - start_time) * 1000,
            model=llm_model(self.config.query.llm),
        ).model_dump()

    async def _execute_index(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute index tool."""
        paths = args["paths"]
        recursive = args.get("recursive", True)
        force = args.get("force", False)

        from ..utils.helpers import read_files

        start_time = time.time()
        all_files: List[Path] = []
        failed_files: List[str] = []

        for path in paths:
            try:
                files = read_files(path, recursive)
                all_files.extend(files)
            except RAGClientError as e:
                failed_files.append(f"{path}: {e}")

        if not all_files:
            return IndexingResult(
                success=False,
                documents_indexed=0,
                nodes_created=0,
                documents_failed=len(failed_files),
                failed_files=failed_files,
                processing_time_ms=(time.time() - start_time) * 1000,
            ).model_dump()

        self.retriever = self.workflow.load_retriever(
            input_files=all_files,
            index_files=force,
            verbose=False,
        )

        return IndexingResult(
            success=True,
            documents_indexed=len(all_files),
            nodes_created=len(all_files),  # Approximate
            documents_failed=len(failed_files),
            failed_files=failed_files,
            processing_time_ms=(time.time() - start_time) * 1000,
        ).model_dump()

    async def _execute_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute status tool."""
        from datetime import datetime
        from ..config.models import embedding_model

        _has_retriever = self.retriever is not None
        _has_query = self.config.query is not None
        _has_chat = self.config.chat is not None

        embed_model = "none"
        embed_dims = 0
        if self.config.retrieval.embedding:
            embed_model = embedding_model(self.config.retrieval.embedding)
            embed_dims = getattr(self.config.retrieval.embedding, "dimensions", 384)

        return IndexStatus(
            total_documents=0,  # Would need to query storage
            total_nodes=0,
            last_updated=datetime.now(),
            embedding_model=embed_model,
            vector_dimensions=embed_dims,
            index_size_mb=0.0,
        ).model_dump()
