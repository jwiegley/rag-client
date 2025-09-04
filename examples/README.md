# RAG Client Examples

This directory contains practical examples demonstrating various use cases of the RAG Client.

## Quick Start Examples

### 1. Basic Document Indexing and Querying

```python
from rag_client.core.workflow import RAGWorkflow
from rag_client.config.loader import load_config

# Load configuration
config = load_config("examples/configs/basic.yaml")

# Initialize workflow
workflow = RAGWorkflow(config)

# Index documents
workflow.index_documents(["/path/to/documents"])

# Query the indexed documents
response = workflow.query("What is machine learning?")
print(response)
```

### 2. Using Different Embedding Providers

```python
# HuggingFace embeddings example
from rag_client.config.models import HuggingFaceEmbeddingConfig

embedding_config = HuggingFaceEmbeddingConfig(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda",
    normalize=True
)

# OpenAI embeddings example
from rag_client.config.models import OpenAIEmbeddingConfig

embedding_config = OpenAIEmbeddingConfig(
    model="text-embedding-ada-002",
    api_key="your-api-key"
)
```

### 3. Chat with Context Memory

```python
from rag_client.core.workflow import RAGWorkflow

workflow = RAGWorkflow(config)

# Start chat session with context memory
chat_engine = workflow.get_chat_engine()

# Multiple turns maintaining context
response1 = chat_engine.chat("What is RAG?")
response2 = chat_engine.chat("How does it improve LLM responses?")
response3 = chat_engine.chat("Can you give me an example?")
```

### 4. API Server Usage

```python
# Start the API server
import uvicorn
from rag_client.api.server import app

uvicorn.run(app, host="0.0.0.0", port=8000)
```

Then use curl or any HTTP client:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "rag-model",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 5. PostgreSQL Vector Store

```python
from rag_client.config.models import VectorStoreConfig

vector_config = VectorStoreConfig(
    type="postgres",
    connection_string="postgresql://user:pass@localhost/ragdb",
    table_name="embeddings",
    embed_dim=768
)

config.retrieval.vector_store = vector_config
workflow = RAGWorkflow(config)
```

## Advanced Examples

### Custom Document Processing

```python
from rag_client.utils.readers import get_file_reader

# Process specific file types
reader = get_file_reader(".pdf")
documents = reader.load_data(file_path="document.pdf")

# Custom chunking
from rag_client.config.models import SplitterConfig

splitter_config = SplitterConfig(
    chunk_size=512,
    chunk_overlap=50,
    separator=" ",
    paragraph_separator="\n\n"
)
```

### Multi-Provider Setup

```python
from rag_client.config.models import Config, RetrievalConfig

# Use different providers for embeddings and LLM
config = Config(
    retrieval=RetrievalConfig(
        embedding=HuggingFaceEmbeddingConfig(
            model_name="BAAI/bge-large-en-v1.5"
        ),
        llm=OllamaConfig(
            model="llama2",
            base_url="http://localhost:11434"
        )
    )
)
```

## Configuration Examples

See the `configs/` directory for complete configuration examples:

- `basic.yaml` - Simple local setup with Ollama
- `openai.yaml` - OpenAI API configuration
- `postgres.yaml` - PostgreSQL vector store setup
- `advanced.yaml` - Multi-provider with custom settings

## Running the Examples

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your configuration file based on examples

3. Run the example scripts:
```bash
python examples/basic_indexing.py
python examples/chat_session.py
python examples/api_server.py
```

## Troubleshooting

### Common Issues

1. **Embedding dimension mismatch**: Ensure your vector store dimension matches the embedding model output
2. **API key errors**: Set environment variables or use config files for API keys
3. **Memory issues**: Adjust batch sizes in configuration
4. **Connection errors**: Verify database/API endpoints are accessible

### Debug Mode

Enable verbose logging:

```python
from rag_client.utils.logging import setup_logging

setup_logging(level="DEBUG", log_file="rag_debug.log")
```

## Further Resources

- [Main Documentation](../docs/index.rst)
- [API Reference](../docs/modules.rst)
- [Configuration Guide](../docs/configuration.rst)