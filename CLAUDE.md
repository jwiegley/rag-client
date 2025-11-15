# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rag-client is a flexible Python tool for Retrieval-Augmented Generation (RAG) that augments LLM interactions with contextual document retrieval. It supports both ephemeral (file-cached) and persistent (Postgres-based) modes, and is designed for integration with local/open-weight models.

## Package Structure

The project is organized as a Python package with a modular architecture:

```
rag_client/                      # Main package directory
├── core/                        # Core RAG functionality
│   ├── workflow.py             # RAGWorkflow class for orchestration
│   ├── models.py               # QueryState and ChatState management
│   ├── retrieval.py            # CustomRetriever implementations
│   └── indexing.py             # Document indexing logic
├── config/                      # Configuration system
│   └── models.py               # Dataclass-based YAML configs
├── cli/                         # Command-line interface
│   ├── __init__.py            # CLI setup and parsing
│   └── commands.py            # Command handlers (index, search, query, chat, serve)
├── api/                         # REST API implementation
│   └── server.py              # FastAPI OpenAI-compatible server
├── providers/                   # LLM and embedding providers
│   └── factory.py             # Provider factory pattern
├── storage/                     # Storage backends
│   ├── ephemeral.py           # File-based caching
│   └── postgres.py            # PostgreSQL with pgvector
├── utils/                       # Helper utilities
│   ├── logging.py             # Centralized logging
│   ├── readers.py             # Custom document readers (OrgReader, MailParser)
│   └── helpers.py             # General utilities
├── exceptions.py               # Custom exception hierarchy
└── types.py                    # Type definitions and aliases

Legacy/Compatibility:
├── main.py                     # CLI entry point
├── rag.py                      # Legacy monolithic implementation (deprecated)
├── rag_compat.py              # Compatibility layer
├── chat.py                     # Custom SimpleContextChatEngine
└── api.py                      # Legacy API server (deprecated)

Configuration Examples:
├── examples/configs/
│   ├── basic.yaml             # Simple Ollama setup
│   ├── openai.yaml            # OpenAI configuration
│   └── postgres.yaml          # Persistent storage setup
├── chat.yaml                  # Main chat configuration
└── guidance.yaml              # Alternative example config

Documentation & Tests:
├── docs/                       # Sphinx documentation
├── tests/                      # Test suite
└── query-test.sh              # Integration test script
```

## Core Components

### Main Entry Point
- **main.py**: CLI argument parsing and command dispatch
  - Uses `typed-argparse` for type-safe argument handling
  - Supports: index, search, query, chat, serve commands
  - Configurable logging levels (--verbose, --debug)

### RAG Workflow (`rag_client/core/workflow.py`)
- **RAGWorkflow** class orchestrates the entire RAG pipeline:
  - `load_config()`: YAML configuration loading with validation
  - `load_retriever()`: Vector index creation and loading
  - `retrieve_nodes()`: Semantic search execution
  - Supports ephemeral (file-cached) and persistent (Postgres) modes
  - Automatic fingerprinting for cache management

### State Management (`rag_client/core/models.py`)
- **QueryState**: Manages query engine and LLM for one-shot Q&A
- **ChatState**: Manages chat engine, memory, and conversation history
- State persistence to `~/.config/rag-client/chat_store.json`

### API Server (`rag_client/api/server.py`)
- OpenAI-compatible FastAPI server with endpoints:
  - `POST /v1/chat/completions`: Chat completions (streaming supported)
  - `POST /v1/completions`: Text completions
  - `POST /v1/embeddings`: Generate embeddings
  - `GET /v1/models`: List available models
- Features: API key auth, CORS middleware, request/response logging

### Custom Components

#### SimpleContextChatEngine (`chat.py`)
- Performance-optimized alternative to standard ContextChatEngine
- Uses COMPACT response mode instead of iterative refinement
- Better throughput for large top_k values
- Limits context assembly to configured window size

#### Custom Readers (`rag_client/utils/readers.py`)
- **OrgReader**: Parses Emacs org-mode files
  - Splits by org entries (headings)
  - Preserves heading and property metadata
  - Body-only text extraction
- **MailParser**: Parses .eml email files
  - Extracts From, Date, Subject headers
  - Plain text body extraction
  - Metadata preservation

## CLI Commands

### Command Reference

```bash
# General format
python main.py --config <yaml-file> [options] <command> [args]
```

#### Global Options
- `--config, -c`: Path to YAML configuration file (required)
- `--from`: Input source (file, directory, or `-` for stdin)
- `--recursive`: Recursively process directories
- `--num-workers, -j`: Parallel workers for document processing
- `--verbose`: Show progress and INFO logs
- `--debug`: Show DEBUG logs and detailed traces
- `--streaming`: Enable streaming responses
- `--top-k`: Number of top chunks to retrieve
- `--sparse-top-k`: Top results for sparse/keyword search
- `--force`: Bypass cache and force re-indexing

#### Commands

**index** - Index documents into vector store:
```bash
# Index from directory
python main.py --config chat.yaml --from /path/to/docs --recursive index

# Index from stdin (file list)
find docs/ -name '*.pdf' | python main.py --config chat.yaml --from - index

# Force re-indexing (ignore cache)
python main.py --config chat.yaml --from docs/ --force index
```

**search** - Retrieve relevant chunks (JSON output):
```bash
python main.py --config chat.yaml --from docs/ search "machine learning"
# Output: [{"text": "...", "metadata": {...}, "score": 0.95}, ...]
```

**query** - Ask questions with LLM-generated answers:
```bash
# Non-streaming
python main.py --config chat.yaml --from docs/ query "Explain transformers"

# Streaming
python main.py --config chat.yaml --from docs/ --streaming query "Explain RAG"
```

**chat** - Interactive conversation mode:
```bash
python main.py --config chat.yaml --from docs/ chat
```
Features:
- Readline history saved to `~/.config/rag-client/chat_history`
- Chat store persisted to `~/.config/rag-client/chat_store.json`
- Inline commands: `search <query>`, `query <query>`, `exit`, `quit`
- History limit: 1000 entries

**serve** - Start OpenAI-compatible API server:
```bash
python main.py --config chat.yaml --from docs/ \
  --host localhost --port 7990 serve

# With auto-reload for development
python main.py --config chat.yaml --from docs/ \
  --host 0.0.0.0 --port 8080 --reload-server serve
```

## Configuration System (YAML)

Configurations use dataclass-wizard for type-safe YAML parsing with union discrimination.

### Complete Configuration Structure

```yaml
# Logging configuration (optional)
logging:
  level: INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: logs/rag.log        # Optional file output
  console_output: true          # Console output
  rotate_logs: true             # Enable log rotation
  max_bytes: 10485760           # 10MB max file size
  backup_count: 5               # Keep 5 rotated files

# Retrieval configuration
retrieval:
  embed_individually: false     # Process files individually then merge

  # Embedding provider (required for indexing/search)
  embedding:
    type: HuggingFaceEmbeddingConfig  # or Ollama, OpenAI, LiteLLM, etc.
    model_name: "BAAI/bge-small-en-v1.5"
    max_length: 512
    normalize: true
    device: "cuda"              # or "cpu", "mps"

  # Vector store (optional, defaults to ephemeral)
  vector_store:
    type: PostgresVectorStoreConfig
    connection: "postgresql://user:pass@host:port/dbname"
    hybrid_search: true         # Enable BM25 + vector search
    dimensions: 384             # Must match embedding model
    hnsw_m: 16                  # HNSW index parameters
    hnsw_ef_construction: 64
    hnsw_ef_search: 40
    hnsw_dist_method: "vector_cosine_ops"

  # Query fusion for multi-strategy retrieval (optional)
  fusion:
    type: FusionRetrieverConfig
    num_queries: 3              # Number of query variations
    mode: "relative_score"      # or "reciprocal_rank"
    llm: *llm                   # LLM for query generation

  # Text splitter (required for indexing)
  splitter:
    type: SentenceSplitterConfig
    chunk_size: 1024
    chunk_overlap: 200
    paragraph_separator: "\n\n\n"

  # Metadata extractors (optional)
  extractors:
    - type: KeywordExtractorConfig
      keywords: 5
      llm: *llm
    - type: SummaryExtractorConfig
      summaries: ["self"]
      llm: *llm

  # Keyword index for hybrid search (optional)
  keywords:
    collect: true
    llm: *llm

# Query configuration (required for 'query' command)
query:
  llm: &llm                     # YAML anchor for reuse
    type: OllamaConfig
    model: "llama3"
    base_url: "http://localhost:11434"
    temperature: 0.7
    keep_alive: "5m"

  engine:
    type: RetrieverQueryEngineConfig
    response_mode: compact_accumulate
    top_k: 10
    sparse_top_k: 10

  retries: false                # Enable retry on poor responses
  source_retries: false         # Enable source-based retry
  show_citations: false         # Include source citations
  evaluator_llm: *llm          # Optional LLM for evaluation

# Chat configuration (required for 'chat' command)
chat:
  llm: *llm                     # Reference to LLM anchor
  default_user: "user"
  summarize: false              # Auto-summarize long histories
  keep_history: true            # Persist chat between sessions

  engine:
    type: SimpleContextChatEngineConfig
    context_window: 32768
    top_k: 10
```

### Provider Configuration Examples

**OpenAI-Like (for local models)**:
```yaml
llm:
  type: OpenAILikeConfig
  model: "Qwen3-235B-Instruct"
  api_base: "http://localhost:8080/v1"
  api_key: "fake"               # or use api_key_command
  api_key_command: "pass api-key | head -1"  # Execute command for key
  context_window: 131072
  temperature: 0.6
  max_tokens: 8192
  timeout: 7200
```

## Exception Hierarchy

The project uses comprehensive exception handling (`rag_client/exceptions.py`):

- **RAGClientError**: Base exception with error codes and context
  - `ConfigurationError`: Invalid configuration
  - `EmbeddingError`: Embedding model failures
  - `LLMError`: LLM provider errors
  - `IndexingError`: Document indexing failures
  - `RetrievalError`: Search failures
  - `StorageError`: Database/storage errors
  - `DocumentProcessingError`: Parsing errors
  - `APIError`: API server errors
  - `ProviderError`: External provider failures
  - `ValidationError`: Input validation errors
  - `RateLimitError`: Rate limiting errors

All exceptions include error codes, contextual data, and cause chaining.

## Development Workflow

### Setup

#### Using Nix (Recommended)
```bash
# Enter development shell
nix develop

# Or use direnv
echo "use flake" > .envrc
direnv allow
```

Nix provides:
- System dependencies: libstdc++, zlib, glib, libpq, openssl
- Development tools: black, basedpyright, isort, autoflake, pylint
- Wrapped binary using `uv` for dependency management

#### Using pip/venv
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Testing

```bash
# Run integration test script
bash query-test.sh --reset --verbose chat

# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=rag_client tests/
```

### Code Quality

```bash
# Format code
black .
isort .

# Type checking
basedpyright .

# Linting
pylint rag_client/
```

### Documentation

```bash
# Build Sphinx documentation
cd docs/
make html
open _build/html/index.html
```

## Provider Support

### Embedding Providers
- HuggingFace (local models)
- Ollama (local models)
- OpenAI / OpenAI-like
- LiteLLM (multi-provider proxy)
- LlamaCPP (GGUF models)

### LLM Providers
- Ollama (local models)
- OpenAI / OpenAI-like (local/remote)
- LiteLLM (multi-provider proxy)
- Perplexity (online models)
- LMStudio (local models)
- OpenRouter (multi-provider)
- LlamaCPP (GGUF models)

## Cache Management

### Ephemeral Mode
- Cache directory: `~/.cache/rag-client/`
- Fingerprinting: file content hashes + config settings
- Automatic cache reuse for unchanged documents
- Use `--force` flag to bypass cache

### Persistent Mode
- PostgreSQL with pgvector extension
- Tables: docstore, indexstore, vectorstore
- HNSW indexing for efficient similarity search
- Supports hybrid search (vector + BM25)

## Common Issues and Solutions

### Configuration Issues

1. **Embedding Dimension Mismatch**:
   - PostgreSQL `dimensions` must match embedding model output
   - Example: `bge-small-en-v1.5` outputs 384 dimensions

2. **API Key Configuration**:
   - Use `api_key_command` for secure key management
   - Example: `api_key_command: "pass my-api-key | head -1"`

3. **YAML Anchor References**:
   - Use `*llm` to reference `&llm` anchor
   - Share LLM config between query and chat

### Performance Optimization

1. **Large top_k Values**:
   - Use `SimpleContextChatEngineConfig` for better performance
   - Avoids iterative refinement overhead

2. **Memory Usage**:
   - Set `embed_individually: true` for large document sets
   - Processes files sequentially, then merges

3. **Parallel Processing**:
   - Use `--num-workers 4` or `-j 4` for faster indexing
   - Optimal value depends on CPU cores

### Provider Issues

1. **Ollama**:
   - Ensure running: `ollama serve`
   - Pull model first: `ollama pull llama3`
   - Exact model names: `ollama list`

2. **Timeout Errors**:
   - Increase `timeout` for slow models
   - Example: `timeout: 7200` (2 hours)

## Important Files

### Configuration Examples
- `examples/configs/basic.yaml`: Simple Ollama setup
- `examples/configs/openai.yaml`: OpenAI configuration
- `examples/configs/postgres.yaml`: Persistent storage
- `chat.yaml`: Main chat configuration
- `guidance.yaml`: Alternative example

### Documentation
- `prd.md`, `prd.txt`: Product requirements
- `validation_report.md`: Code quality report
- `README.md`: User-facing documentation
- `docs/`: Sphinx documentation source

### Example Data
- `RAG-LlamaIndex/`: Example implementations and tutorials
- Various test scripts: `main_test.py`, `test_rag.py`, `client.py`

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md