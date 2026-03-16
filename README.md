# rag-client

I've been running local LLMs for a while now, and one thing that kept
frustrating me was the gap between "chat with a model" and "chat with a model
about my actual documents." I didn't want to upload anything to a third party,
and I didn't want to cobble together a different retrieval pipeline every time I
had a new set of files to work with. So I built rag-client: a Python tool that
handles the whole RAG workflow -- indexing, search, querying, chat, and even an
OpenAI-compatible API server -- driven by a single YAML config file.

It's built on top of [llama-index](https://www.llamaindex.ai/) and works with
local models through Ollama, llama.cpp, LM Studio, and anything that speaks the
OpenAI API format. There's also support for OpenAI, LiteLLM, OpenRouter, and
Perplexity if you want to use hosted models. For storage, you can keep things
ephemeral (file-cached, no database needed) or persistent (Postgres with
pgvector).

I won't pretend the project is polished to a shine -- there are rough edges,
and the configuration has more knobs than it probably should -- but it does what
I need it to do, and it's been quite handy for my own Emacs/GPTel workflow.

## Getting started

The easiest way to get a working environment is with Nix:

```bash
nix develop
```

This drops you into a shell with Python, all system dependencies (libpq,
openssl, zlib, etc.), and development tools. It uses `uv` under the hood for
Python dependency management, so you don't need to run `pip install` yourself.

If you'd rather use a plain venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Everything is driven by a YAML config file. Here's a minimal example using
Ollama for both embeddings and the LLM:

```yaml
retrieval:
  embedding:
    type: OllamaEmbeddingConfig
    model_name: nomic-embed-text

  splitter:
    type: SentenceSplitterConfig
    chunk_size: 512
    chunk_overlap: 20

query:
  llm: &llm
    type: OllamaConfig
    model: llama3
    base_url: "http://localhost:11434"
    temperature: 0.7

  engine:
    type: RetrieverQueryEngineConfig
    response_mode: compact_accumulate

chat:
  llm: *llm
  engine:
    type: SimpleContextChatEngineConfig
    context_window: 32768
```

The `&llm` / `*llm` YAML anchors let you share the same LLM config between
query and chat without repeating yourself. There are example configs under
`examples/configs/` for different setups (OpenAI, Postgres, etc.), and
`chat.yaml` in the repo root is the one I actually use day-to-day.

The config system supports a wide range of providers -- HuggingFace embeddings,
OpenAI-like endpoints, LlamaCPP with GGUF models, and more. The type
discriminator field (`type: ...`) tells the config parser which provider to
instantiate.

## CLI commands

The general pattern is:

```bash
python main.py --config <yaml-file> --from <source> [options] <command>
```

Where `--from` can be a file, a directory, or `-` to read a file list from
stdin.

### index

Index documents into the vector store:

```bash
python main.py -c chat.yaml --from ~/docs --recursive index
```

Use `--force` to bypass the cache and re-index from scratch. The cache lives in
`~/.cache/rag-client/` and uses content hashing -- so if your documents haven't
changed, re-indexing is a no-op.

### search

Retrieve matching chunks as JSON (no LLM involved):

```bash
python main.py -c chat.yaml --from ~/docs search "memory management"
```

### query

Ask a question and get an LLM-generated answer grounded in your documents:

```bash
python main.py -c chat.yaml --from ~/docs query "How does the garbage collector work?"
```

Add `--streaming` for streamed output.

### chat

Interactive conversation with retrieval context:

```bash
python main.py -c chat.yaml --from ~/docs chat
```

This gives you a readline-enabled prompt with history (saved to
`~/.config/rag-client/chat_history`). You can type `search <query>` or
`query <query>` inline, or just chat normally.

### serve

Start an OpenAI-compatible API server:

```bash
python main.py -c chat.yaml --from ~/docs --host localhost --port 7990 serve
```

This exposes `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and
`/v1/models` -- so you can point any OpenAI-compatible client at it and get
RAG-augmented responses. I use this with GPTel in Emacs.

## Useful options

- `--verbose` / `--debug`: Control log verbosity
- `--top-k N`: Number of chunks to retrieve (default depends on config)
- `--sparse-top-k N`: Top results for keyword/BM25 search
- `--num-workers N` / `-j N`: Parallel workers for document processing
- `--recursive`: Process directories recursively
- `--streaming`: Stream LLM responses

## Persistent storage with Postgres

By default, rag-client uses ephemeral file-cached storage. If you want a
persistent vector store, add a `vector_store` section to your retrieval config:

```yaml
retrieval:
  vector_store:
    type: PostgresVectorStoreConfig
    connection: "postgresql://user:pass@localhost:5432/mydb"
    hybrid_search: true
    dimensions: 384  # must match your embedding model
```

This uses pgvector for similarity search and optionally supports hybrid
(vector + BM25) retrieval. Make sure the `dimensions` value matches what your
embedding model actually produces -- a mismatch here is a common source of
confusing errors.

## Development

### Environment

`nix develop` is the recommended path. It gives you everything you need,
including ruff, basedpyright, and pytest.

### Pre-commit hooks

The project uses [lefthook](https://github.com/evilmartians/lefthook) for
pre-commit hooks. These run automatically on `git commit` and check:

- **ruff format** and **ruff lint** on Python files
- **shfmt** on shell scripts
- **nix flake check** for Nix-level validation
- **pytest** on the config and fuzz test suites
- **benchmarks** against a saved baseline (if one exists)

### Running tests

```bash
pytest tests/

# With coverage
pytest --cov=rag_client tests/
```

### Code quality

```bash
ruff format .
ruff check .
basedpyright .
```

## License

BSD 3-Clause. See [LICENSE.md](LICENSE.md) for the full text.
