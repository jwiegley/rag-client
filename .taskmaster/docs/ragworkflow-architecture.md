# RAGWorkflow Class Architecture Documentation

## Overview
The `RAGWorkflow` class is the core orchestrator of the RAG pipeline, located in `rag.py` (lines 1397-2547). It's a dataclass that manages the entire lifecycle of document ingestion, indexing, retrieval, and query/chat operations.

## Class Definition

```python
@dataclass
class RAGWorkflow:
    logger: logging.Logger
    config: Config
```

### Core Attributes
- **logger**: Python logging instance for workflow operations
- **config**: Main configuration object containing all settings (retrieval, query, chat)

## Method Categories and Architecture

### 1. Configuration Loading
#### `load_config(cls, path: Path) -> Config` (Line 1403)
- **Purpose**: Load YAML configuration file
- **Type**: Class method
- **Flow**: Read YAML → Validate as Config object → Return or error
- **Error Handling**: Exits on invalid/missing config file

### 2. Component Loading Infrastructure

#### Private Component Loaders
##### `__load_component(cls, config, verbose, component_type)` (Line 1452)
- **Purpose**: Factory method for loading embeddings or LLMs
- **Type**: Private class method
- **Supports**: 6 embedding providers, 9 LLM providers
- **Pattern**: Match-case statement for provider selection
- **Returns**: BaseEmbedding or LLM instance

##### `__load_embedding(cls, config, verbose)` (Line 1559)
- **Purpose**: Load embedding model with validation
- **Wrapper for**: `__load_component` with type checking

##### `__load_llm(cls, config, verbose)` (Line 1571)  
- **Purpose**: Load LLM with validation
- **Wrapper for**: `__load_component` with type checking

##### `realize_llm(cls, config, verbose)` (Line 1583)
- **Purpose**: Public interface for LLM instantiation
- **Error Handling**: Exits if LLM creation fails

### 3. Document Processing Pipeline

#### Document Reading
##### `__read_documents(cls, input_files, num_workers, recursive, verbose)` (Line 1615)
- **Purpose**: Load documents from filesystem
- **Custom Readers**: 
  - `.org` → OrgReader
  - `.eml` → MailParser
- **Default**: SimpleDirectoryReader for other formats
- **Returns**: Iterable[Document]

#### Document Processing
##### `__process_documents(self, documents, embedding, keywords, splitter, extractors, verbose)` (Line 1715)
- **Purpose**: Transform documents into nodes
- **Pipeline Steps**:
  1. Apply splitter (if configured)
  2. Apply extractors (if configured)
  3. Collect keywords (if configured)
  4. Generate embeddings
- **Returns**: List of processed nodes

#### Splitter Loading
##### `__load_splitter(cls, splitter, verbose)` (Line 1633)
- **Supports**: 5 splitter types
  - SentenceSplitter
  - SentenceWindowNodeParser
  - SemanticSplitterNodeParser
  - JSONNodeParser
  - CodeSplitter

#### Extractor Loading
##### `__load_extractor(cls, config, verbose)` (Line 1680)
- **Supports**: 4 extractor types
  - KeywordExtractor
  - SummaryExtractor
  - TitleExtractor
  - QuestionsAnsweredExtractor

### 4. Storage Management

#### PostgreSQL Storage
##### `__postgres_stores(cls, config, embedding_dimensions)` (Line 1416)
- **Purpose**: Initialize PostgreSQL storage components
- **Creates**:
  - PostgresDocumentStore (table: docstore)
  - PostgresIndexStore (table: indexstore)
  - PGVectorStore (table: vectorstore)
- **Configuration**: HNSW indexing parameters

#### Storage Context Loading
##### `__load_storage_context(self, embedding, store_config, verbose)` (Line 1803)
- **Purpose**: Create or load storage context
- **Modes**:
  - SimpleVectorStore (in-memory)
  - PostgresVectorStore (persistent)
- **Returns**: StorageContext instance

### 5. Index Management

#### Index Building
##### `__build_vector_index(self, nodes, storage_context, service_context, verbose)` (Line 1757)
- **Purpose**: Create vector index from nodes
- **Index Types**:
  - VectorStoreIndex (default)
  - SummaryIndex
  - TreeIndex
- **Features**: Progress tracking, node insertion

#### Index Persistence
##### `__persist_dir(self, input_files)` (Line 1798)
- **Purpose**: Determine cache directory for index
- **Location**: `~/.cache/rag/<fingerprint>`

##### `__save_indices(self, indices, storage_context, persist_dir)` (Line 1873)
- **Purpose**: Save indices to disk
- **Format**: JSON metadata + binary data

##### `__load_indices(self, embedding, store_config, persist_dir, verbose)` (Line 1885)
- **Purpose**: Load existing indices
- **Fallback**: Disk → Cache → None

### 6. Ingestion Workflow

#### Main Ingestion
##### `__ingest_files(self, input_files, embedding, retrieval, verbose)` (Line 1971)
- **Purpose**: Complete document ingestion pipeline
- **Steps**:
  1. Check for existing index (cache/storage)
  2. Read documents
  3. Process documents
  4. Build indices
  5. Save indices
- **Optimization**: Fingerprint-based caching

##### `__ingest_documents(self, documents, embedding, retrieval, verbose)` (Line 1836)
- **Purpose**: Process document list into indices
- **Internal**: Called by `__ingest_files`

#### Fingerprinting
##### `__determine_fingerprint(cls, input_files, config)` (Line 1602)
- **Purpose**: Create unique cache key
- **Inputs**: File hashes + config representation
- **Output**: 32-character base64 string

### 7. Retrieval System

#### Retriever Loading
##### `load_retriever(self, input_files, config, verbose)` (Line 2095)
- **Purpose**: Create retriever from indices
- **Retriever Types**:
  - VectorIndexRetriever
  - BM25Retriever (keyword)
  - QueryFusionRetriever (multi-query)
- **Features**: Top-k configuration, reranking

#### Node Retrieval
##### `retrieve_nodes(self, query, retriever, top_k, verbose)` (Line 2164)
- **Purpose**: Execute retrieval query
- **Returns**: Ranked nodes with scores

#### Retriever Creation
##### `__retriever_from_index(self, index, keywords, top_k, verbose)` (Line 2016)
- **Purpose**: Create retriever from index
- **Modes**:
  - Vector-only
  - Keyword-only
  - Hybrid (vector + keyword)

#### Node Merging
##### `__merge_nodes(cls, vector_nodes, sparse_nodes, top_k, verbose)` (Line 2083)
- **Purpose**: Combine vector and keyword results
- **Strategy**: Score-based ranking

### 8. Query Processing

#### Query Initialization
##### `__init__(self, input_files, config, verbose)` (Line 2181)
- **Purpose**: Initialize query engine
- **Creates**: Retriever, LLM, query engine

#### Query Engine Loading
##### `__load_query_engine(cls, engine_config, index, retriever, llm, llm_config, evaluator_llm, verbose)` (Line 2232)
- **Purpose**: Create query engine with configuration
- **Engine Types**:
  - SimpleQueryEngine
  - CitationQueryEngine
  - RetrieverQueryEngine
  - MultiStepQueryEngine
  - RetryQueryEngine
  - RetrySourceQueryEngine

##### `__load_base_query_engine(cls, config, index, retriever, llm, llm_config, verbose)` (Line 2198)
- **Purpose**: Create base query engine
- **Handles**: Response synthesis, streaming

#### Query Execution
##### `query(self, query)` (Line 2300)
- **Purpose**: Execute query and return response
- **Returns**: RESPONSE_TYPE (streaming or complete)

#### Evaluator Loading
##### `__load_evaluator(cls, config, verbose)` (Line 2284)
- **Purpose**: Create response evaluator
- **Types**:
  - RelevancyEvaluator
  - GuidelineEvaluator

### 9. Chat System

#### Chat Initialization
##### `__init__(self, input_files, config, keep_history, verbose)` (Line 2313)
- **Purpose**: Initialize chat engine
- **Components**: Retriever, LLM, memory, chat engine

#### Chat Execution
##### `chat(self, message, user, chat_history)` (Line 2399)
- **Purpose**: Process chat message
- **Features**: Memory management, context retrieval
- **Returns**: StreamingChatResponse or ChatResponse

## Workflow Lifecycle

### 1. Indexing Phase
```
Input Files → Read Documents → Process Documents → Build Index → Save Index
```

### 2. Retrieval Phase
```
Load Index → Create Retriever → Configure Fusion/Reranking
```

### 3. Query Phase
```
Query → Retrieve Nodes → Synthesize Response → Return Result
```

### 4. Chat Phase
```
Message → Update Memory → Retrieve Context → Generate Response → Update History
```

## Storage Strategies

### Ephemeral Storage
- **Location**: `~/.cache/rag/<fingerprint>/`
- **Components**: SimpleVectorStore, SimpleDocumentStore, SimpleIndexStore
- **Persistence**: File-based JSON/pickle

### Persistent Storage (PostgreSQL)
- **Tables**: docstore, indexstore, vectorstore
- **Features**: pgvector extension, HNSW indexing
- **Configuration**: Connection string, dimensions, search parameters

## Key Design Patterns

### 1. Factory Pattern
- Component loading via `__load_component`
- Provider selection through match-case

### 2. Builder Pattern
- Index construction with multiple configuration options
- Query engine assembly with decorators

### 3. Strategy Pattern
- Multiple retrieval strategies (vector, keyword, hybrid)
- Various response synthesis modes

### 4. Template Method
- Document processing pipeline with pluggable components
- Ingestion workflow with caching logic

## Dependencies and Integration Points

### External Services
- **Embedding Providers**: HuggingFace, OpenAI, Ollama, etc.
- **LLM Providers**: OpenAI, Perplexity, LMStudio, etc.
- **Storage**: PostgreSQL with pgvector

### Internal Components
- **Config**: Complete configuration management
- **Custom Readers**: OrgReader, MailParser
- **Custom Retrievers**: ColoredKeywordRetriever

## Error Handling

### Fatal Errors (exit)
- Configuration loading failures
- LLM initialization failures
- Missing required components

### Graceful Degradation
- Cache miss → rebuild index
- Storage unavailable → fallback to memory

## Performance Optimizations

### 1. Caching Strategy
- Fingerprint-based index caching
- Avoids redundant processing

### 2. Batch Processing
- Embedding generation in batches
- Document processing with workers

### 3. Lazy Loading
- Indices loaded on demand
- Components initialized when needed

## Method Statistics

- **Total Methods**: 38
- **Class Methods**: 20 (52.6%)
- **Instance Methods**: 18 (47.4%)
- **Public Methods**: 6
- **Private Methods**: 32 (prefixed with __)

## Recommendations for Refactoring

1. **Separate Concerns**: Split into multiple focused classes
   - IndexManager
   - RetrieverFactory
   - QueryProcessor
   - ChatEngine

2. **Reduce Method Complexity**: Break down large methods
   - `__ingest_files` (45 lines)
   - `__load_component` (100+ lines)

3. **Improve Error Handling**: Replace exits with exceptions

4. **Standardize Patterns**: Consistent factory implementation

5. **Extract Constants**: Move magic strings to configuration