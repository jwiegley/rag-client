"""MCP (Model Context Protocol) server for RAG client.

This module provides MCP-compatible tool definitions for AI agent integration,
allowing agents like Claude, GPT, and others to use RAG capabilities.
"""

from .server import MCPServer, create_mcp_server
from .tools import RAGTools

__all__ = [
    "MCPServer",
    "RAGTools",
    "create_mcp_server",
]
