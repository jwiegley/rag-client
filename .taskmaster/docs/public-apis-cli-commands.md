# Public APIs and CLI Commands Documentation

## CLI Commands

### Global Arguments
All commands support these global arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config`, `-c` | string | **required** | YAML config file path |
| `--from` | string | optional | Where to read files from |
| `--recursive` | flag | false | Read directories recursively |
| `--num-workers`, `-j` | int | optional | Number of parallel jobs |
| `--verbose` | flag | false | Enable verbose output |
| `--debug` | flag | false | Enable debug logging |
| `--streaming` | flag | false | Enable streaming responses |
| `--top-k` | int | optional | Top number of chunks to return |
| `--sparse-top-k` | int | optional | Top number for sparse retrieval |

### Command: `index`
Indexes documents into the vector store.

**Usage:**
```bash
python main.py --config config.yaml index [files...]
```

**Behavior:**
- Reads files from `--from` directory or individual file paths
- Processes documents through the ingestion pipeline
- Creates embeddings and stores in configured backend
- Supports recursive directory reading with `--recursive`
- Can parallelize with `--num-workers`

**Example:**
```bash
python main.py --config chat.yaml --from ./documents --recursive index
```

### Command: `search`
Performs vector similarity search on indexed documents.

**Usage:**
```bash
python main.py --config config.yaml search "query text"
```

**Returns:**
- JSON array of retrieved nodes with scores
- Each node contains text content and metadata

**Example:**
```bash
python main.py --config chat.yaml --top-k 5 search "machine learning concepts"
```

**Output Format:**
```json
[
  {
    "node": {
      "text": "...",
      "metadata": {...}
    },
    "score": 0.85
  }
]
```

### Command: `query`
Executes a RAG query with LLM response generation.

**Usage:**
```bash
python main.py --config config.yaml query "question"
```

**Behavior:**
- Retrieves relevant context using configured retriever
- Generates response using configured LLM
- Supports streaming with `--streaming` flag
- Requires `query` section in config

**Example:**
```bash
python main.py --config chat.yaml --streaming query "What is the capital of France?"
```

### Command: `chat`
Starts an interactive chat session with context retrieval.

**Usage:**
```bash
python main.py --config config.yaml chat
```

**Features:**
- Interactive REPL interface
- Maintains conversation history
- Supports inline commands:
  - `search <query>` - Perform search within chat
  - `query <question>` - Execute query within chat
  - `exit` or `quit` - Exit chat session
- Saves history to `~/.config/rag-client/chat_history`
- Optional persistent chat store

**Example Session:**
```
user> Hello, how can you help me?
assistant> I can help you search documents and answer questions...
user> search machine learning
[search results displayed]
user> What did you find about neural networks?
assistant> Based on the documents...
user> exit
Goodbye!
```

### Command: `serve`
Starts the OpenAI-compatible API server.

**Usage:**
```bash
python main.py --config config.yaml serve
```

**Server Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | localhost | Server host address |
| `--port` | 7990 | Server port |
| `--reload-server` | false | Auto-reload on code changes |

**Example:**
```bash
python main.py --config chat.yaml --host 0.0.0.0 --port 8080 serve
```

## REST API Endpoints

### Authentication
All API endpoints support optional authentication via Bearer token:
```
Authorization: Bearer sk-test
```

The expected key is read from `API_KEY` environment variable (default: "sk-test").

### POST `/v1/chat/completions`
OpenAI-compatible chat completion endpoint.

**Request Body:**
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": false,
  "max_tokens": 150,
  "temperature": 0.7
}
```

**Response (Non-streaming):**
```json
{
  "id": "chatcmpl-123456",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

**Response (Streaming):**
Server-Sent Events format:
```
data: {"id":"chatcmpl-123456","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"chatcmpl-123456","object":"chat.completion.chunk","choices":[{"delta":{"content":"!"}}]}
data: {"id":"chatcmpl-123456","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

### POST `/v1/completions`
OpenAI-compatible text completion endpoint.

**Request Body:**
```json
{
  "model": "text-davinci-003",
  "prompt": "Once upon a time",
  "max_tokens": 50,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**
```json
{
  "id": "cmpl-123456",
  "object": "text_completion",
  "created": 1234567890,
  "model": "text-davinci-003",
  "choices": [
    {
      "text": "...generated text...",
      "index": 0,
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 20,
    "total_tokens": 24
  }
}
```

**Note:** Currently returns empty responses - implementation incomplete.

### POST `/v1/embeddings`
Generate embeddings for input text.

**Request Body:**
```json
{
  "model": "text-embedding-ada-002",
  "input": "Sample text to embed"
}
```

Or with array input:
```json
{
  "model": "text-embedding-ada-002",
  "input": ["Text 1", "Text 2", "Text 3"]
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

**Note:** Currently returns dummy embeddings - needs implementation.

### GET `/v1/models`
List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.1-8b-RAG",
      "object": "model",
      "created": 1234567890,
      "owned_by": "your-organization"
    }
  ]
}
```

The model ID is derived from the configured LLM or embedding model with "-RAG" suffix.

### GET `/`
Health check endpoint.

**Response:**
```json
{
  "status": "API is running",
  "version": "1.0.0"
}
```

## Error Handling

All endpoints return standard HTTP error responses:

**401 Unauthorized:**
```json
{
  "detail": "Invalid API key"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Error message describing the issue"
}
```

## Configuration Requirements

### For `query` command:
Config must include a `query` section with:
- `llm`: LLM configuration
- `response_mode`: How to synthesize responses
- Query engine type configuration

### For `chat` command:
Config must include a `chat` section with:
- `llm`: LLM configuration
- `engine_type`: Chat engine type
- `default_user`: User name (default: "user")
- `keep_history`: Whether to persist chat history

### For `serve` command:
Requires both `retrieval` and either `chat` or `query` sections configured.

## State Management

### Chat State
- Maintains conversation context
- Stores chat history if configured
- Persists to `~/.config/rag-client/chat_store.json`

### Query State
- Manages query engine and retriever
- Handles response synthesis
- Supports streaming responses

### API Server State
- Global workflow instance
- Shared retriever instance
- Lazy-initialized query and chat states

## Usage Examples

### Complete Indexing and Query Workflow
```bash
# Index documents
python main.py --config config.yaml --from ./docs --recursive index

# Search for content
python main.py --config config.yaml --top-k 10 search "machine learning"

# Ask a question
python main.py --config config.yaml query "What are the key concepts?"

# Start interactive chat
python main.py --config config.yaml chat
```

### API Server Usage
```bash
# Start server
python main.py --config config.yaml --host 0.0.0.0 --port 8080 serve

# Use with curl
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-test" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Integration with OpenAI Python Client
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7990/v1",
    api_key="sk-test"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Notes and Limitations

1. **Embeddings endpoint** currently returns placeholder data
2. **Completions endpoint** is not fully implemented
3. **Token counting** in API responses uses estimates
4. **Authentication** is basic and should be enhanced for production
5. **Error handling** could be more granular
6. **Streaming** works for chat but not completions

## Recommended Improvements

1. Implement actual embedding generation in `/v1/embeddings`
2. Complete the `/v1/completions` implementation
3. Add proper token counting using tiktoken
4. Enhance authentication with JWT or OAuth
5. Add rate limiting and request validation
6. Implement model management endpoints
7. Add metrics and monitoring endpoints
8. Support for function calling in chat completions