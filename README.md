# rag-client: Retrieval Augmentation for LLM Queries

## Overview

`rag-client.py` is a flexible Python tool for augmenting LLM interactions with contextual document retrieval through two modes: **Ephemeral** (in-memory) and **Persistent** (Postgres-based). It integrates Retrieval-Augmented Generation (RAG) to ground LLM responses in your documents while maintaining compatibility with local/open-weight models.

## Why Use This Tool?

- **Flexible RAG Modes**: Choose between disposable sessions or long-term knowledge bases
- **Emacs/GPTel Integration**: Designed for seamless use within Emacs workflows
- **Local-First Philosophy**: Works with self-hosted LLMs (e.g., Falcon3-10B) and embedding models
- **Performance Tuning**: Granular control over HNSW indexes, chunking, and search parameters

## Key Features

| Category              | Capabilities                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| **Ephemeral Mode**    | On-demand indexing, content-hashed cache, multi-query session optimization |
| **Persistent Mode**   | PGVector storage, reusable document sets, table-based collection management|
| **Embedding Control** | Custom chunk sizes/overlap, HuggingFace models, GPU acceleration           |
| **LLM Compatibility** | OpenAI API format support, local model integration, timeout/retry handling |

## Installation

```
pip install -r requirements.txt  # Includes llama-index, pgvector, sqlalchemy
```

## Modes of Operation

### 🌀 Ephemeral Mode (In-Memory)

*Ideal for*: One-off research sessions, temporary document analysis

```
echo "/path/to/docs/*.md" | python rag-client.py \
  --from -                                       \
  --embed-provider "HuggingFace"                 \
  --embed-model "BAAI/bge-large-en-v1.5"         \
  --chunk-size 512                               \
  --chunk-overlap 20                             \
  --top-k 15                                     \
  search "Keyword analysis"
```

- Auto-caches index to `~/.cache/rag-client` using file content hashes
- Subsequent queries in same session skip re-indexing

### 💾 Persistent Mode (Postgres)

*Ideal for*: Enterprise knowledge bases, frequent document set reuse

1. **Initial Indexing**:

```
find /storage/pdfs -name '*.pdf' | python rag-client.py \
  --db-name research_papers                             \
  --db-table ai_ethics                                  \
  --embed-provider "HuggingFace"                        \
  --embed-model "BAAI/bge-large-en-v1.5"                \
  --chunk-size 512                                      \
  --chunk-overlap 20                                    \
  --from -                                              \
  index
```

2. **Querying**:

```
python rag-client.py                        \
  --db-name research_papers                 \
  --db-table ai_ethics                      \
  --llm-provider "OpenAILike"               \
  --llm "Falcon3-10B-Instruct"              \
  --llm-base-url "http://localhost:8080/v1" \
  --timeout 600                             \
  --top-k 25                                \
  query "Compare AI safety approaches"
```

3. **Chat sessions**:

Simply remove the `--search` or `--query` options:

```
python rag-client.py                        \
  --db-name research_papers                 \
  --db-table ai_ethics                      \
  --llm-provider "OpenAILike"               \
  --llm "Falcon3-10B-Instruct"              \
  --llm-base-url "http://localhost:8080/v1" \
  --timeout 600                             \
  --top-k 25                                \
  chat
```

3. **Present an OpenAI-compatible API**:

Simply remove the `--search` or `--query` options:

```
python rag-client.py                        \
  --db-name research_papers                 \
  --db-table ai_ethics                      \
  --llm-provider "OpenAILike"               \
  --llm "Falcon3-10B-Instruct"              \
  --llm-base-url "http://localhost:8080/v1" \
  --timeout 600                             \
  --top-k 25                                \
  --host localhost                          \
  --port 9090                               \
  serve
```

## Configuration Highlights

Use `rag-client --help` for the full list of parameters and their default
values.

### Critical Parameters

#### Embedding

```
--embed-provider=PROVIDER
--embed-model=MODEL
--embed-dim=DIMENSIONS
--chunk-size=CHUNKING_SIZE
--chunk-overlap=OVERLAP_SIZE
--top-k=RETURN_TOP_K_RESULTS
```

#### Database

The database parameter indicates “persistent” use, and is not required.

```
--db-name=NAME        # Name of the Postgres+Pgvector database
```

#### LLM

The LLM parameter is only needed for chat queries, and not if you are only
embedding and searching a document collection.

```
--llm-provider=PROVIDER
--llm=MODEL
--llm-api-key=API_KEY_IF_NEEDED
--llm-base-url=http://localhost:8080/v1  # Local model endpoint
--timeout=TIMEOUT_IN_SECS
--temperature=TEMPERATURE
--context-window=CONTEXT_WINDOW
```

## GPTel Integration

This is work in progress, and depends on a [draft PR branch of
gptel](https://github.com/jwiegley/gptel/tree/johnw/augment) combined with a
[new gptel-rag module](https://github.com/jwiegley/dot-emacs/blob/master/lisp/gptel-rag.el).

## Troubleshooting

**Common Issues**:

- Embedding Dimension Mismatch: Ensure `--embed-dim` matches embedding model
- Chunk Overflows: Reduce `--chunk-size` for dense technical content

**Debug Flags**:

- `--verbose`: Inspect index construction timing
- `--token-limit=500`: Adjust for long-form generation
- `--timeout=60`: Adjust for long-running local models
