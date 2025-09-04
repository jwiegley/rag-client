# Provider Implementations Documentation

## Overview
The rag-client supports 6 embedding providers and 9 LLM providers through a unified factory pattern implemented in the `RAGWorkflow.__load_component` method (lines 1452-1557 in rag.py).

## Embedding Provider Implementations

### 1. HuggingFace Embeddings
**Configuration Class**: `HuggingFaceEmbeddingConfig` (Line 268)
**Implementation**: Lines 1459-1463

```python
case HuggingFaceEmbeddingConfig():
    return HuggingFaceEmbedding(
        **asdict(config),
        show_progress_bar=verbose,
    )
```

**Key Features**:
- Local model execution
- Device selection (CPU/GPU)
- Parallel processing support
- Custom instructions for query/text
- Model caching

**Configuration Requirements**:
- `model_name`: Default "BAAI/bge-small-en-v1.5"
- `max_length`: Optional token limit
- `device`: Optional device specification
- `cache_folder`: Local model cache
- `trust_remote_code`: Security flag

### 2. Ollama Embeddings
**Configuration Class**: `OllamaEmbeddingConfig` (Line 283)
**Implementation**: Lines 1464-1468

```python
case OllamaEmbeddingConfig():
    return OllamaEmbedding(
        **asdict(config),
        show_progress=verbose,
    )
```

**Key Features**:
- Local server integration
- Custom base URL support
- Additional kwargs passthrough

**Configuration Requirements**:
- `model_name`: Required (e.g., "nomic-embed-text")
- `base_url`: Default "http://localhost:11434"
- `embed_batch_size`: Default batch size

### 3. OpenAI Embeddings
**Configuration Class**: `OpenAIEmbeddingConfig` (Line 292)
**Implementation**: Lines 1474-1485

```python
case OpenAIEmbeddingConfig():
    if config.api_key_command is not None:
        config.api_key = subprocess.run(
            config.api_key_command,
            shell=True,
            text=True,
            capture_output=True,
        ).stdout.rstrip("\n")
    return OpenAIEmbedding(
        **asdict(config),
        show_progress=verbose,
    )
```

**Key Features**:
- API key command execution
- Multiple embedding modes
- Dimension configuration
- Retry logic with exponential backoff

**Configuration Requirements**:
- `model`: Default "text-embedding-ada-002"
- `api_key`: Required (direct or via command)
- `dimensions`: Optional for ada-003
- `max_retries`: Default 10

### 4. OpenAI-Like Embeddings
**Configuration Class**: `OpenAILikeEmbeddingConfig` (Line 310)
**Implementation**: Lines 1486-1513

```python
case OpenAILikeEmbeddingConfig():
    # API key command execution
    # LiteLLM session ID and logging configuration
    extra_body = {}
    if config.add_litellm_session_id:
        extra_body["litellm_session_id"] = str(uuid.uuid1())
    if config.no_litellm_logging:
        extra_body["no-log"] = True
    # Additional kwargs merging
    return OpenAILikeEmbedding(
        show_progress=verbose,
        **asdict(config),
    )
```

**Key Features**:
- OpenAI-compatible API support
- LiteLLM integration
- Session tracking
- Custom API endpoints

**Configuration Requirements**:
- `model_name`: Required
- `api_base`: Custom endpoint URL
- `api_key`: Default "fake" for local models

### 5. LiteLLM Embeddings
**Configuration Class**: `LiteLLMEmbeddingConfig` (Line 329)
**Implementation**: Lines 1514-1524

```python
case LiteLLMEmbeddingConfig():
    if config.api_key_command is not None:
        config.api_key = subprocess.run(
            config.api_key_command,
            shell=True,
            text=True,
            capture_output=True,
        ).stdout.rstrip("\n")
    return LiteLLMEmbedding(
        **asdict(config),
    )
```

**Key Features**:
- Multi-provider proxy
- Unified interface for multiple providers
- API key command support

**Configuration Requirements**:
- `model_name`: Provider-prefixed model
- `api_key`: Provider-specific key

### 6. LlamaCPP Embeddings
**Configuration Class**: `LlamaCPPEmbeddingConfig` (Line 346)
**Implementation**: Lines 1469-1473

```python
case LlamaCPPEmbeddingConfig():
    return LlamaCPPEmbedding(
        **asdict(config),
        show_progress=verbose,
    )
```

**Key Features**:
- Local GGUF model execution
- GPU layer offloading
- Memory mapping
- Extensive performance tuning

**Configuration Requirements**:
- `model_path`: Path to GGUF file
- `n_gpu_layers`: GPU acceleration
- `n_ctx`: Context window size
- 38 total configuration options

## LLM Provider Implementations

### 1. Ollama LLM
**Configuration Class**: `OllamaConfig` (Line 426)
**Implementation**: Lines 1525-1526

```python
case OllamaConfig():
    return Ollama(**asdict(config), show_progress=verbose)
```

**Key Features**:
- Local model serving
- Function calling support
- Keep-alive configuration
- JSON mode

**Configuration Requirements**:
- `model`: Required model name
- `base_url`: Default "http://localhost:11434"
- `temperature`: Default 0.75
- `context_window`: Default context size

### 2. OpenAI LLM
**Configuration Class**: `OpenAIConfig` (Line 440)
**Implementation**: Lines 1545-1546

```python
case OpenAIConfig():
    return OpenAI(**asdict(config), show_progress=verbose)
```

**Key Features**:
- GPT model support
- Streaming responses
- Function calling
- Structured output (strict mode)
- Audio configuration

**Configuration Requirements**:
- `model`: Default "gpt-3.5-turbo"
- `api_key`: Required
- `temperature`: Default temperature
- `max_tokens`: Optional limit

### 3. OpenAI-Like LLM
**Configuration Class**: `OpenAILikeConfig` (Line 463)
**Implementation**: Lines 1527-1542

```python
case OpenAILikeConfig():
    # LiteLLM session and logging configuration
    extra_body = {}
    # ... configuration setup
    return OpenAILike(**asdict(config))
```

**Key Features**:
- Compatible with OpenAI API clones
- Configurable chat/function calling modes
- LiteLLM integration

**Inherits from**: OpenAIConfig
**Additional Fields**:
- `context_window`: Model context size
- `is_chat_model`: Chat capability flag
- `is_function_calling_model`: Function calling flag

### 4. LiteLLM LLM
**Configuration Class**: `LiteLLMConfig` (Line 472)
**Implementation**: Lines 1543-1544

```python
case LiteLLMConfig():
    return LiteLLM(**asdict(config))
```

**Key Features**:
- Multi-provider proxy
- Unified interface across providers
- Provider routing

**Inherits from**: OpenAIConfig
**Configuration**: Similar to OpenAILike

### 5. LlamaCPP LLM
**Configuration Class**: `LlamaCPPConfig` (Line 479)
**Implementation**: Lines 1547-1548

```python
case LlamaCPPConfig():
    return LlamaCPP(**asdict(config))
```

**Key Features**:
- Local GGUF model execution
- URL or file path loading
- Custom generation parameters

**Configuration Requirements**:
- `model_url` or `model_path`: Model location
- `temperature`: Generation temperature
- `max_new_tokens`: Output limit
- `context_window`: Context size

### 6. Perplexity LLM
**Configuration Class**: `PerplexityConfig` (Line 493)
**Implementation**: Lines 1549-1550

```python
case PerplexityConfig():
    return Perplexity(**asdict(config), show_progress=verbose)
```

**Key Features**:
- Online search integration
- Search classifier
- Optimized for factual responses

**Configuration Requirements**:
- `model`: Default "sonar-pro"
- `api_key`: Required
- `temperature`: Default 0.2 (lower for accuracy)
- `enable_search_classifier`: Search control

### 7. OpenRouter LLM
**Configuration Class**: `OpenRouterConfig` (Line 509)
**Implementation**: Lines 1551-1552

```python
case OpenRouterConfig():
    return OpenRouter(**asdict(config), show_progress=verbose)
```

**Key Features**:
- Multi-model routing
- Cost optimization
- Model selection

**Configuration Requirements**:
- `model`: Model identifier
- `api_key`: Required
- `api_base`: Default API endpoint

### 8. LMStudio LLM
**Configuration Class**: `LMStudioConfig` (Line 521)
**Implementation**: Lines 1553-1554

```python
case LMStudioConfig():
    return LMStudio(**asdict(config))
```

**Key Features**:
- Local server integration
- OpenAI-compatible API
- Custom model management

**Configuration Requirements**:
- `model_name`: Required
- `base_url`: Default "http://localhost:1234/v1"
- `is_chat_model`: Default true

### 9. MLX LLM
**Configuration Class**: `MLXLLMConfig` (Line 536)
**Implementation**: Lines 1555-1556

```python
case MLXLLMConfig():
    return MLXLLM(**asdict(config))
```

**Key Features**:
- Apple Silicon optimization
- Local model execution
- Custom tokenizer support

**Configuration Requirements**:
- `model_name`: Default MLX model
- `context_window`: Context size
- `max_new_tokens`: Output limit

## Provider Initialization Patterns

### Common Patterns

1. **API Key Management**
```python
if config.api_key_command is not None:
    config.api_key = subprocess.run(
        config.api_key_command,
        shell=True,
        text=True,
        capture_output=True,
    ).stdout.rstrip("\n")
```
Used by: OpenAI, OpenAI-Like, LiteLLM

2. **Verbose Progress Display**
```python
return Provider(
    **asdict(config),
    show_progress=verbose,  # or show_progress_bar
)
```
Used by: Most providers

3. **Extra Body Configuration**
```python
extra_body = {}
if config.add_litellm_session_id:
    extra_body["litellm_session_id"] = str(uuid.uuid1())
if config.no_litellm_logging:
    extra_body["no-log"] = True
```
Used by: OpenAI-Like providers

4. **Configuration Spreading**
```python
return Provider(**asdict(config))
```
All providers use dataclass to dict conversion

## Provider Compatibility Matrix

| Feature | HuggingFace | Ollama | OpenAI | LiteLLM | LlamaCPP | Perplexity |
|---------|-------------|---------|---------|----------|-----------|------------|
| **Embeddings** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **LLM** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Local** | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| **API Key** | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| **Streaming** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Function Calling** | ❌ | ✅ | ✅ | Varies | ❌ | ❌ |
| **Batch Processing** | ✅ | ✅ | ✅ | ✅ | Limited | ✅ |

## Provider Selection Logic

The `RAGWorkflow.__load_component` method uses pattern matching:

```python
def __load_component(
    cls,
    config: EmbeddingConfig | LLMConfig,
    verbose: bool = False,
    component_type: Literal["embedding", "llm"] = "embedding",
) -> BaseEmbedding | LLM | NoReturn:
    match config:
        case HuggingFaceEmbeddingConfig():
            # ... initialization
        case OllamaConfig():
            # ... initialization
        # ... other cases
```

### Error Handling
- Invalid configuration exits the program
- Missing required fields cause initialization failure
- API key failures are not gracefully handled

## Code Duplication Issues

### Identified Duplications

1. **API Key Command Execution** (3 occurrences)
   - Lines 1475-1481 (OpenAI)
   - Lines 1487-1493 (OpenAI-Like)
   - Lines 1515-1521 (LiteLLM)

2. **Extra Body Configuration** (2 occurrences)
   - Lines 1495-1508 (OpenAI-Like Embedding)
   - Lines 1528-1541 (OpenAI-Like LLM)

3. **Progress Display Pattern** (15+ occurrences)
   - Same pattern across all providers

### Refactoring Opportunities

1. **Extract API Key Resolution**
```python
def resolve_api_key(config):
    if hasattr(config, 'api_key_command') and config.api_key_command:
        return subprocess.run(...).stdout.rstrip("\n")
    return config.api_key
```

2. **Extract Extra Body Builder**
```python
def build_extra_body(config):
    extra_body = {}
    if getattr(config, 'add_litellm_session_id', False):
        extra_body["litellm_session_id"] = str(uuid.uuid1())
    # ... other configurations
    return extra_body
```

3. **Provider Factory Registry**
```python
PROVIDER_REGISTRY = {
    HuggingFaceEmbeddingConfig: lambda c, v: HuggingFaceEmbedding(...),
    OllamaEmbeddingConfig: lambda c, v: OllamaEmbedding(...),
    # ... other providers
}
```

## Provider Initialization Workflow

```
Configuration Object
    ↓
Pattern Matching (match/case)
    ↓
Provider-Specific Setup
    ├── API Key Resolution
    ├── Extra Configuration
    └── Verbose Settings
    ↓
Provider Instantiation
    ↓
Return Provider Instance
```

## Recommendations

1. **Reduce Code Duplication**
   - Extract common initialization patterns
   - Create base provider initialization class

2. **Improve Error Handling**
   - Replace exits with exceptions
   - Add validation before initialization

3. **Standardize Configuration**
   - Create base configuration classes
   - Inherit common fields

4. **Lazy Loading**
   - Load providers only when needed
   - Cache initialized providers

5. **Provider Registry Pattern**
   - Replace match/case with registry
   - Enable dynamic provider registration