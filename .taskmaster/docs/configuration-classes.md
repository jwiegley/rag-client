# Configuration Classes Documentation

## Overview
The rag-client codebase uses YAMLWizard (dataclass-wizard) for configuration management with 60+ configuration dataclasses. All configuration classes inherit from YAMLWizard for YAML serialization/deserialization.

## Configuration Class Hierarchy

### Main Configuration Class
```python
@dataclass
class Config(YAMLWizard):  # Line 824
    retrieval: RetrievalConfig
    query: QueryConfig | None = None
    chat: ChatConfig | None = None
```

## Embedding Configuration Classes (6 Classes)

### 1. HuggingFaceEmbeddingConfig (Line 268)
```python
@dataclass
class HuggingFaceEmbeddingConfig(YAMLWizard):
    model_name: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
    max_length: int | None = None
    query_instruction: str | None = None
    text_instruction: str | None = None
    normalize: bool = True
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    cache_folder: str | None = None
    trust_remote_code: bool = False
    device: str | None = None
    parallel_process: bool = False
    target_devices: list[str] | None = None
```

### 2. OllamaEmbeddingConfig (Line 283)
```python
@dataclass
class OllamaEmbeddingConfig(YAMLWizard):
    model_name: str
    base_url: str = "http://localhost:11434"
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    ollama_additional_kwargs: dict[str, Any] | None = None
    client_kwargs: dict[str, Any] | None = None
```

### 3. OpenAIEmbeddingConfig (Line 292)
```python
@dataclass
class OpenAIEmbeddingConfig(YAMLWizard):
    mode: str = OpenAIEmbeddingMode.TEXT_SEARCH_MODE
    model: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
    embed_batch_size: int = 100
    dimensions: int | None = None
    api_key: str | None = None
    api_base: str | None = None
    # ... (15 total fields)
```

### 4. OpenAILikeEmbeddingConfig (Line 310)
```python
@dataclass
class OpenAILikeEmbeddingConfig(YAMLWizard):
    model_name: str
    embed_batch_size: int = 10
    api_key: str = "fake"
    api_base: str | None = None
    # ... (14 total fields)
```

### 5. LiteLLMEmbeddingConfig (Line 329)
```python
@dataclass
class LiteLLMEmbeddingConfig(YAMLWizard):
    model_name: str
    embed_batch_size: int = 10
    api_key: str = "fake"
    # ... (13 total fields)
```

### 6. LlamaCPPEmbeddingConfig (Line 346)
```python
@dataclass
class LlamaCPPEmbeddingConfig(YAMLWizard):
    model_path: Path
    n_gpu_layers: int = 0
    # ... (38 total fields - most complex embedding config)
```

**Type Alias**: `EmbeddingConfig` (Line 399) - Union of all embedding configs

## LLM Configuration Classes (9 Classes)

### 1. OllamaConfig (Line 426)
```python
@dataclass
class OllamaConfig(YAMLWizard):
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.75
    context_window: int = DEFAULT_CONTEXT_WINDOW
    # ... (9 total fields)
```

### 2. OpenAIConfig (Line 440)
```python
@dataclass
class OpenAIConfig(YAMLWizard):
    model: str = DEFAULT_OPENAI_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int | None = None
    api_key: str = "fake"
    # ... (19 total fields)
```

### 3. OpenAILikeConfig (Line 463)
Inherits from OpenAIConfig, adds:
```python
@dataclass
class OpenAILikeConfig(OpenAIConfig):
    context_window: int = DEFAULT_CONTEXT_WINDOW
    is_chat_model: bool = False
    is_function_calling_model: bool = False
    add_litellm_session_id: bool = False
    no_litellm_logging: bool = False
```

### 4. LiteLLMConfig (Line 472)
Inherits from OpenAIConfig, adds:
```python
@dataclass
class LiteLLMConfig(OpenAIConfig):
    context_window: int = DEFAULT_CONTEXT_WINDOW
    is_chat_model: bool = False
    is_function_calling_model: bool = False
```

### 5. LlamaCPPConfig (Line 479)
```python
@dataclass
class LlamaCPPConfig(YAMLWizard):
    model_url: str | None = None
    model_path: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    # ... (9 total fields)
```

### 6. PerplexityConfig (Line 493)
```python
@dataclass
class PerplexityConfig(YAMLWizard):
    model: str = "sonar-pro"
    temperature: float = 0.2
    api_key: str | None = None
    # ... (12 total fields)
```

### 7. OpenRouterConfig (Line 509)
```python
@dataclass
class OpenRouterConfig(YAMLWizard):
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    # ... (8 total fields)
```

### 8. LMStudioConfig (Line 521)
```python
@dataclass
class LMStudioConfig(YAMLWizard):
    model_name: str
    base_url: str = "http://localhost:1234/v1"
    # ... (10 total fields)
```

### 9. MLXLLMConfig (Line 536)
```python
@dataclass
class MLXLLMConfig(YAMLWizard):
    context_window: int = DEFAULT_CONTEXT_WINDOW
    model_name: str = DEFAULT_MLX_MODEL
    # ... (11 total fields)
```

**Type Alias**: `LLMConfig` (Line 551) - Union of all LLM configs

## Splitter Configuration Classes (5 Classes)

### 1. SentenceSplitterConfig (Line 600)
```python
@dataclass
class SentenceSplitterConfig(YAMLWizard):
    chunk_size: int = 512
    chunk_overlap: int = 20
    include_metadata: bool = True
```

### 2. SentenceWindowSplitterConfig (Line 607)
```python
@dataclass
class SentenceWindowSplitterConfig(YAMLWizard):
    window_size: int = 3
    window_metadata_key: str = "window"
    original_text_metadata_key: str = "original_text"
```

### 3. SemanticSplitterConfig (Line 614)
```python
@dataclass
class SemanticSplitterConfig(YAMLWizard):
    embedding: EmbeddingConfig | None
    buffer_size: int = 256
    breakpoint_percentile_threshold: int = 95
    include_metadata: bool = True
```

### 4. JSONNodeParserConfig (Line 622)
```python
@dataclass
class JSONNodeParserConfig(YAMLWizard):
    include_metadata: bool = True
    include_prev_next_rel: bool = True
```

### 5. CodeSplitterConfig (Line 628)
```python
@dataclass
class CodeSplitterConfig(YAMLWizard):
    language: str = "python"
    chunk_lines: int = 40
    chunk_lines_overlap: int = 15
    max_chars: int = 1500
```

**Type Alias**: `SplitterConfig` (Line 635) - Union of all splitter configs

## Extractor Configuration Classes (4 Classes)

### 1. KeywordExtractorConfig (Line 645)
```python
@dataclass
class KeywordExtractorConfig(YAMLWizard):
    llm: LLMConfig
    keywords: int = 5
```

### 2. SummaryExtractorConfig (Line 651)
```python
@dataclass
class SummaryExtractorConfig(YAMLWizard):
    llm: LLMConfig
    summaries: list[str] | None = None
```

### 3. TitleExtractorConfig (Line 659)
```python
@dataclass
class TitleExtractorConfig(YAMLWizard):
    llm: LLMConfig
    nodes: int = 5
```

### 4. QuestionsAnsweredExtractorConfig (Line 665)
```python
@dataclass
class QuestionsAnsweredExtractorConfig(YAMLWizard):
    llm: LLMConfig
    questions: int = 1
```

**Type Alias**: `ExtractorConfig` (Line 670) - Union of all extractor configs

## Evaluator Configuration Classes (2 Classes)

### 1. RelevancyEvaluatorConfig (Line 679)
```python
@dataclass
class RelevancyEvaluatorConfig(YAMLWizard):
    llm: LLMConfig
```

### 2. GuidelineConfig (Line 684)
```python
@dataclass
class GuidelineConfig(YAMLWizard):
    llm: LLMConfig
    guidelines: str = DEFAULT_GUIDELINES
```

**Type Alias**: `EvaluatorConfig` (Line 689) - Union of evaluator configs

## Query Engine Configuration Classes (7 Classes)

### Base Query Engine Configs (3 Classes)

1. **SimpleQueryEngineConfig** (Line 704) - Empty config
2. **CitationQueryEngineConfig** (Line 693)
   - chunk_size: int = 512
   - chunk_overlap: int = 20
3. **RetrieverQueryEngineConfig** (Line 699)
   - response_mode: ResponseMode = ResponseMode.REFINE

### Advanced Query Engine Configs (4 Classes)

4. **MultiStepQueryEngineConfig** (Line 714)
   - engine: BaseQueryEngineConfig
5. **RetrySourceQueryEngineConfig** (Line 719)
   - llm: LLMConfig
   - evaluator: EvaluatorConfig
   - engine: BaseQueryEngineConfig
6. **RetryQueryEngineConfig** (Line 726)
   - evaluator: EvaluatorConfig
   - engine: BaseQueryEngineConfig

**Type Aliases**:
- `BaseQueryEngineConfig` (Line 708) - Union of simple/citation/retriever configs
- `QueryEngineConfig` (Line 731) - Union of all query engine configs

## Chat Engine Configuration Classes (4 Classes)

### 1. SimpleChatEngineConfig (Line 740)
Empty configuration class

### 2. SimpleContextChatEngineConfig (Line 745)
```python
@dataclass
class SimpleContextChatEngineConfig(YAMLWizard):
    context_window: int = DEFAULT_CONTEXT_WINDOW
```

### 3. ContextChatEngineConfig (Line 750)
Empty configuration class

### 4. CondensePlusContextChatEngineConfig (Line 755)
```python
@dataclass
class CondensePlusContextChatEngineConfig(YAMLWizard):
    skip_condense: bool = False
```

**Type Alias**: `ChatEngineConfig` (Line 759) - Union of all chat engine configs

## Vector Store Configuration Classes (2 Classes)

### 1. SimpleVectorStoreConfig (Line 768)
Empty configuration class

### 2. PostgresVectorStoreConfig (Line 773)
```python
@dataclass
class PostgresVectorStoreConfig(YAMLWizard):
    connection: str
    hybrid_search: bool = False
    dimensions: int = 512
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    hnsw_dist_method: str = "vector_cosine_ops"
```

**Type Alias**: `VectorStoreConfig` (Line 783) - Union of vector store configs

## Other Configuration Classes

### KeywordsConfig (Line 594)
```python
@dataclass
class KeywordsConfig(YAMLWizard):
    collect: bool = False
    llm: LLMConfig | None = None
```

### FusionRetrieverConfig (Line 787)
```python
@dataclass
class FusionRetrieverConfig(YAMLWizard):
    llm: LLMConfig
    num_queries: int = 1
    mode: FUSION_MODES = FUSION_MODES.RELATIVE_SCORE
```

### RetrievalConfig (Line 794)
```python
@dataclass
class RetrievalConfig(YAMLWizard):
    embed_individually: bool = False
    embedding: EmbeddingConfig | None = None
    keywords: KeywordsConfig | None = None
    splitter: SplitterConfig | None = None
    extractors: list[ExtractorConfig] | None = None
    vector_store: VectorStoreConfig | None = None
    fusion: FusionRetrieverConfig | None = None
```

### QueryConfig (Line 805)
```python
@dataclass
class QueryConfig(YAMLWizard):
    llm: LLMConfig
    engine: QueryEngineConfig | None = None
    retries: bool = False
    source_retries: bool = False
    show_citations: bool = False
    evaluator_llm: LLMConfig | None = None
```

### ChatConfig (Line 815)
```python
@dataclass
class ChatConfig(YAMLWizard):
    llm: LLMConfig
    engine: ChatEngineConfig | None = None
    default_user: str = "user"
    summarize: bool = False
    keep_history: bool = False
```

## Non-YAMLWizard Dataclasses

### EmbeddedNode (Line 1365)
```python
@dataclass
class EmbeddedNode(JSONWizard):
    node: TextNode
    is_enabled: bool = True
```
Uses JSONWizard instead of YAMLWizard

### GlobalJSONMeta (Line 588)
```python
@final
class GlobalJSONMeta(JSONWizard.Meta):
    tag_key = "type"
    auto_assign_tags = True
```

## Configuration Type Unions

The codebase uses TypeAlias extensively to create union types:

1. **EmbeddingConfig** - Union of 6 embedding configs
2. **LLMConfig** - Union of 9 LLM configs
3. **SplitterConfig** - Union of 5 splitter configs
4. **ExtractorConfig** - Union of 4 extractor configs
5. **EvaluatorConfig** - Union of 2 evaluator configs
6. **QueryEngineConfig** - Union of 7 query engine configs
7. **ChatEngineConfig** - Union of 4 chat engine configs
8. **VectorStoreConfig** - Union of 2 vector store configs

## Configuration Inheritance Pattern

```
YAMLWizard (base class from dataclass-wizard)
    ├── All primary configuration classes
    └── Config (main configuration container)

OpenAIConfig
    ├── OpenAILikeConfig
    └── LiteLLMConfig
```

## Usage Patterns

### 1. YAML File Loading
```python
config = Config.from_yaml_file("config.yaml")
```

### 2. Nested Configuration
The main `Config` class contains:
- `retrieval` (RetrievalConfig) - Required
- `query` (QueryConfig) - Optional
- `chat` (ChatConfig) - Optional

### 3. Provider Selection
Helper functions extract model names:
- `embedding_model(config: EmbeddingConfig) -> str` (Line 409)
- `llm_model(config: LLMConfig) -> str` (Line 564)

## Key Observations

1. **Heavy Use of Optional Fields**: Most configs use Optional types with None defaults
2. **API Key Management**: Multiple patterns for API keys (direct, command-based)
3. **Provider-Specific Kwargs**: Most providers support additional_kwargs dictionaries
4. **Default Values**: Extensive use of module-level constants for defaults
5. **Complex Nesting**: Deep configuration nesting (e.g., Config → RetrievalConfig → EmbeddingConfig)

## Statistics

- **Total Configuration Classes**: 42 YAMLWizard classes + 1 JSONWizard class
- **Embedding Configs**: 6 classes
- **LLM Configs**: 9 classes  
- **Splitter Configs**: 5 classes
- **Extractor Configs**: 4 classes
- **Query Engine Configs**: 7 classes
- **Chat Engine Configs**: 4 classes
- **Other Configs**: 7 classes
- **Average Fields per Class**: ~7 fields
- **Most Complex Class**: LlamaCPPEmbeddingConfig (38 fields)

## Refactoring Recommendations

1. **Replace YAMLWizard with Pydantic** for better validation
2. **Reduce configuration complexity** by using composition over deep nesting
3. **Standardize API key handling** across all providers
4. **Extract common fields** into base classes (e.g., temperature, context_window)
5. **Separate concerns** - move configurations to dedicated modules