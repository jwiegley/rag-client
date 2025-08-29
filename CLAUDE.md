# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rag-client is a flexible Python tool for Retrieval-Augmented Generation (RAG) that augments LLM interactions with contextual document retrieval. It supports both ephemeral (in-memory) and persistent (Postgres-based) modes, and is designed for integration with local/open-weight models.

## Core Components

- **main.py**: Entry point that handles CLI argument parsing and command dispatch
- **rag.py**: Core RAG workflow implementation with indexing, retrieval, and query processing  
- **api.py**: OpenAI-compatible API server implementation using FastAPI
- **chat.py**: Chat-related dataclasses and configuration structures

## Development Commands

### Running the Application

```bash
# Direct Python execution
python main.py --config <yaml-file> <command> [args]
```

### Testing

```bash
# Run tests with example data
bash query-test.sh --reset --verbose chat
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Architecture

The system follows a modular architecture:

1. **Configuration Layer**: YAML-based config files define parameters for embeddings, LLMs, database connections, and chunking strategies
2. **Storage Layer**: Supports both ephemeral (file-based caching) and persistent (Postgres with pgvector) storage
3. **Processing Layer**: Uses llama-index for document ingestion, embedding generation, and retrieval
4. **Interface Layer**: Provides CLI commands (index, search, query, chat, serve) and OpenAI-compatible API

Key workflows:
- **Indexing**: Documents → Chunking → Embedding → Storage (cache or database)
- **Retrieval**: Query → Embedding → Vector Search → Reranking → Context Assembly
- **Generation**: Retrieved Context + Query → LLM → Response

## Configuration System

The system uses YAML configuration files (e.g., `chat.yaml`, `guidance.yaml`) to define:
- Embedding provider settings (HuggingFace, OpenAI, Ollama, etc.)
- LLM configuration (provider, model, base URL, temperature)
- Chunking parameters (size, overlap)
- Database connection details
- Retrieval settings (top_k, sparse_top_k)

## Dependencies

The project uses llama-index as its core framework with various provider extensions:
- Embedding providers: HuggingFace, OpenAI, Ollama, LiteLLM
- LLM providers: OpenAI, Ollama, Perplexity, LMStudio, OpenRouter
- Vector stores: Postgres (pgvector)
- Document processing: pypdf, orgparse, tree-sitter

## Nix Development

The project includes Nix flake support for reproducible development environments:
- `flake.nix`: Defines development shell and package build
- `default.nix`: Package derivation
- Provides isolated Python environment with system dependencies

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
