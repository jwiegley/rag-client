"""MCP server implementation for RAG client.

This module provides a Model Context Protocol (MCP) compatible server
that exposes RAG capabilities as tools for AI agents.

The server can run in two modes:
1. Stdio mode: For direct integration with AI assistants
2. HTTP mode: For network-accessible tool serving

Example:
    # Start MCP server in stdio mode
    python -m rag_client.mcp.server --config chat.yaml

    # Start MCP server in HTTP mode
    python -m rag_client.mcp.server --config chat.yaml --http --port 8080
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from ..config.models import Config
from ..core.workflow import RAGWorkflow
from ..utils.logging import get_logger, setup_logging
from .tools import RAGTools

logger = get_logger(__name__)


class MCPServer:
    """Model Context Protocol server for RAG tools.

    Implements the MCP specification for tool discovery and execution,
    allowing AI agents to use RAG capabilities.

    Attributes:
        workflow: RAGWorkflow instance
        tools: RAGTools instance with tool implementations

    Example:
        >>> server = MCPServer(workflow, retriever)
        >>> await server.run_stdio()
    """

    PROTOCOL_VERSION = "2024-11-05"
    SERVER_NAME = "rag-client"
    SERVER_VERSION = "0.1.0"

    def __init__(
        self,
        workflow: RAGWorkflow,
        retriever: Any | None = None,
        config: Config | None = None,
    ):
        """Initialize MCP server.

        Args:
            workflow: RAGWorkflow instance
            retriever: Optional pre-loaded retriever
            config: Optional configuration override
        """
        self.workflow = workflow
        self.config = config or workflow.config
        self.tools = RAGTools(workflow, retriever, config)
        self._initialized = False

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Handle incoming MCP message.

        Args:
            message: MCP protocol message

        Returns:
            Response message
        """
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})

        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = await self._handle_list_tools(params)
            elif method == "tools/call":
                result = await self._handle_call_tool(params)
            elif method == "ping":
                result = {}
            else:
                return self._error_response(msg_id, -32601, f"Unknown method: {method}")

            return self._success_response(msg_id, result)

        except Exception as e:
            logger.error(f"Error handling {method}: {e}")
            return self._error_response(msg_id, -32603, str(e))

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True

        return {
            "protocolVersion": self.PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": self.SERVER_NAME,
                "version": self.SERVER_VERSION,
            },
        }

    async def _handle_list_tools(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/list request."""
        tools = [tool.to_dict() for tool in self.tools.tool_definitions]
        return {"tools": tools}

    async def _handle_call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        result = await self.tools.execute(tool_name, arguments)

        if result.get("success"):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result["result"], indent=2, default=str),
                    }
                ],
            }
        else:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {result.get('error', 'Unknown error')}",
                    }
                ],
                "isError": True,
            }

    def _success_response(self, msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        """Create success response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result,
        }

    def _error_response(self, msg_id: Any, code: int, message: str) -> dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": code,
                "message": message,
            },
        }

    async def run_stdio(self) -> None:
        """Run MCP server in stdio mode.

        Reads JSON-RPC messages from stdin and writes responses to stdout.
        This is the standard mode for AI assistant integration.
        """
        logger.info("Starting MCP server in stdio mode")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        (
            writer_transport,
            writer_protocol,
        ) = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(
            writer_transport, writer_protocol, reader, asyncio.get_event_loop()
        )

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                message = json.loads(line.decode())
                response = await self.handle_message(message)

                response_bytes = (json.dumps(response) + "\n").encode()
                writer.write(response_bytes)
                await writer.drain()

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.error(f"Error in stdio loop: {e}")

    async def run_http(self, host: str = "localhost", port: int = 8080) -> None:
        """Run MCP server in HTTP mode.

        Provides HTTP endpoints for tool discovery and execution.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError("aiohttp required for HTTP mode: pip install aiohttp")

        async def handle_mcp(request: web.Request) -> web.Response:
            try:
                message = await request.json()
                response = await self.handle_message(message)
                return web.json_response(response)
            except Exception as e:
                return web.json_response(
                    self._error_response(None, -32700, str(e)), status=400
                )

        async def handle_tools(request: web.Request) -> web.Response:
            tools = [tool.to_dict() for tool in self.tools.tool_definitions]
            return web.json_response({"tools": tools})

        async def handle_health(request: web.Request) -> web.Response:
            return web.json_response({"status": "ok", "server": self.SERVER_NAME})

        app = web.Application()
        app.router.add_post("/mcp", handle_mcp)
        app.router.add_get("/tools", handle_tools)
        app.router.add_get("/health", handle_health)

        logger.info(f"Starting MCP HTTP server on {host}:{port}")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        # Keep running
        while True:
            await asyncio.sleep(3600)


def create_mcp_server(
    config_path: Path | None = None,
    input_from: str | None = None,
) -> MCPServer:
    """Create an MCP server instance.

    Args:
        config_path: Path to YAML config file
        input_from: Optional path to pre-index

    Returns:
        Configured MCPServer instance
    """
    from ..cli.simple import create_default_config
    from ..utils.helpers import read_files

    if config_path:
        config = RAGWorkflow.load_config(config_path)
    else:
        config = create_default_config()

    logger_inst = get_logger("mcp")
    workflow = RAGWorkflow(logger_inst, config)

    retriever = None
    if input_from:
        input_files = read_files(input_from, recursive=True)
        retriever = workflow.load_retriever(input_files=input_files)

    return MCPServer(workflow, retriever, config)


async def main() -> None:
    """Main entry point for MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Client MCP Server")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--from",
        dest="input_from",
        type=str,
        help="Path to index on startup",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run in HTTP mode instead of stdio",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="HTTP host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port (default: 8080)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "WARNING"
    setup_logging(level=log_level, colored=True)

    config_path = Path(args.config) if args.config else None
    server = create_mcp_server(config_path, args.input_from)

    if args.http:
        await server.run_http(args.host, args.port)
    else:
        await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
