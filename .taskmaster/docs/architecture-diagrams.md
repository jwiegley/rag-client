# System Architecture and Data Flow Documentation

## High-Level System Architecture

```mermaid
graph TB
    subgraph "User Interfaces"
        CLI[CLI Commands]
        API[REST API Server]
    end
    
    subgraph "Core Application Layer"
        MAIN[main.py<br/>Entry Point]
        RAG[RAGWorkflow<br/>Orchestrator]
        APIMOD[api.py<br/>FastAPI Server]
    end
    
    subgraph "Configuration Layer"
        YAML[YAML Config Files]
        CONFIG[Configuration Classes<br/>42 YAMLWizard Classes]
        CHAT[chat.py<br/>Config Definitions]
    end
    
    subgraph "Provider Layer"
        subgraph "Embedding Providers"
            HF[HuggingFace]
            OAI_E[OpenAI]
            OLL_E[Ollama]
            LITE_E[LiteLLM]
            CPP_E[LlamaCPP]
            OAIL_E[OpenAI-Like]
        end
        
        subgraph "LLM Providers"
            OLL_L[Ollama]
            OAI_L[OpenAI]
            PERP[Perplexity]
            LMS[LMStudio]
            OR[OpenRouter]
            CPP_L[LlamaCPP]
            LITE_L[LiteLLM]
            MLX[MLX]
            OAIL_L[OpenAI-Like]
        end
    end
    
    subgraph "Processing Layer"
        ING[Ingestion Pipeline]
        SPLIT[Node Parsers<br/>5 Types]
        EXTRACT[Extractors<br/>4 Types]
        INDEX[Indexing]
        RETR[Retrieval]
        SYNTH[Response Synthesis]
    end
    
    subgraph "Storage Layer"
        subgraph "Ephemeral"
            CACHE[File Cache<br/>XDG Cache]
            MEM[In-Memory Stores]
        end
        
        subgraph "Persistent"
            PG[PostgreSQL<br/>pgvector]
            PGDOC[Postgres Doc Store]
            PGIDX[Postgres Index Store]
        end
    end
    
    CLI --> MAIN
    API --> APIMOD
    MAIN --> RAG
    APIMOD --> RAG
    YAML --> CONFIG
    CONFIG --> CHAT
    CHAT --> RAG
    RAG --> HF & OAI_E & OLL_E & LITE_E & CPP_E & OAIL_E
    RAG --> OLL_L & OAI_L & PERP & LMS & OR & CPP_L & LITE_L & MLX & OAIL_L
    RAG --> ING --> SPLIT --> EXTRACT --> INDEX
    INDEX --> CACHE & MEM
    INDEX --> PG & PGDOC & PGIDX
    RETR --> INDEX
    SYNTH --> RETR
```

## Data Flow Through RAG Pipeline

### 1. Document Indexing Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant RAG as RAGWorkflow
    participant Reader as Document Reader
    participant Parser as Node Parser
    participant Embed as Embedding Provider
    participant Store as Vector Store
    
    User->>CLI: index command
    CLI->>RAG: initialize(config)
    RAG->>Reader: read_files(paths)
    Reader-->>RAG: Documents[]
    RAG->>Parser: parse_documents()
    Parser-->>RAG: Nodes[]
    RAG->>Embed: get_embeddings(nodes)
    Embed-->>RAG: Embeddings[]
    RAG->>Store: store_vectors(embeddings)
    Store-->>RAG: success
    RAG-->>User: Index complete
```

### 2. Query Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant RAG as RAGWorkflow
    participant Retriever
    participant Embed as Embedding Provider
    participant Store as Vector Store
    participant LLM
    participant Synth as Response Synthesizer
    
    User->>CLI: query "question"
    CLI->>RAG: query(text)
    RAG->>Embed: embed_query(text)
    Embed-->>RAG: query_embedding
    RAG->>Retriever: retrieve(embedding)
    Retriever->>Store: similarity_search()
    Store-->>Retriever: NodeWithScore[]
    Retriever-->>RAG: relevant_nodes
    RAG->>Synth: synthesize(query, nodes)
    Synth->>LLM: generate(prompt)
    LLM-->>Synth: response
    Synth-->>RAG: final_response
    RAG-->>User: Answer
```

### 3. Chat Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant Chat as Chat Loop
    participant State as ChatState
    participant Memory as Chat Memory
    participant Engine as Chat Engine
    participant Retriever
    participant LLM
    
    User->>Chat: message
    Chat->>State: initialize()
    State->>Memory: load_history()
    Memory-->>State: conversation[]
    State->>Engine: chat(message)
    Engine->>Retriever: get_context()
    Retriever-->>Engine: relevant_docs
    Engine->>LLM: generate(context+message)
    LLM-->>Engine: response
    Engine->>Memory: save(message, response)
    Engine-->>State: response
    State-->>User: Display response
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "Configuration Management"
        YW[YAMLWizard Base]
        EC[Embedding Configs<br/>6 Types]
        LC[LLM Configs<br/>9 Types]
        SC[Splitter Configs<br/>5 Types]
        EXC[Extractor Configs<br/>4 Types]
        QC[Query Configs<br/>7 Types]
        CC[Chat Configs<br/>4 Types]
        
        YW --> EC & LC & SC & EXC & QC & CC
    end
    
    subgraph "Provider Factory"
        PF[Provider Factory<br/>__load_component()]
        EP[Embedding Providers]
        LP[LLM Providers]
        
        EC --> PF
        LC --> PF
        PF --> EP
        PF --> LP
    end
    
    subgraph "Core Workflow"
        RW[RAGWorkflow]
        QS[QueryState]
        CS[ChatState]
        
        RW --> QS
        RW --> CS
        EP --> RW
        LP --> RW
    end
```

## Storage Layer Architecture

```mermaid
graph TB
    subgraph "Storage Strategy Selection"
        CONFIG[Config.storage_strategy]
        CONFIG -->|ephemeral| EPH[Ephemeral Storage]
        CONFIG -->|persistent| PERS[Persistent Storage]
    end
    
    subgraph "Ephemeral Storage"
        EPH --> CACHE_DIR[XDG Cache Directory]
        EPH --> SIMPLE_VEC[SimpleVectorStore]
        EPH --> SIMPLE_DOC[SimpleDocumentStore]
        EPH --> SIMPLE_IDX[SimpleIndexStore]
        
        CACHE_DIR --> |serialize| JSON[JSON Files]
        CACHE_DIR --> |hash-based| DEDUP[Deduplication]
    end
    
    subgraph "Persistent Storage"
        PERS --> PG_CONN[PostgreSQL Connection]
        PG_CONN --> PGVEC[PGVectorStore<br/>pgvector extension]
        PG_CONN --> PG_DOC[PostgresDocumentStore]
        PG_CONN --> PG_IDX[PostgresIndexStore]
        
        PGVEC --> |similarity| COSINE[Cosine Distance]
        PGVEC --> |indexing| IVFFLAT[IVFFlat Index]
    end
```

## Main Workflow Sequence Diagrams

### Index Command Workflow

```mermaid
stateDiagram-v2
    [*] --> ParseArgs: User runs index command
    ParseArgs --> LoadConfig: Read YAML config
    LoadConfig --> InitRAG: Create RAGWorkflow
    InitRAG --> ReadFiles: Get input files
    ReadFiles --> CheckCache: Hash-based dedup
    CheckCache --> AlreadyIndexed: File exists in cache
    CheckCache --> ProcessNew: New file
    ProcessNew --> ParseDocs: Node parsing
    ParseDocs --> ExtractMeta: Run extractors
    ExtractMeta --> GenerateEmbed: Create embeddings
    GenerateEmbed --> StoreVectors: Save to storage
    StoreVectors --> UpdateCache: Update hash cache
    AlreadyIndexed --> UpdateCache
    UpdateCache --> [*]: Complete
```

### API Server Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Auth as Auth Middleware
    participant Handler as Request Handler
    participant RAG as RAGWorkflow
    participant State as Global State
    
    Client->>FastAPI: POST /v1/chat/completions
    FastAPI->>Auth: verify_api_key()
    Auth-->>FastAPI: authorized
    FastAPI->>Handler: create_chat_completion()
    Handler->>State: get chat_state
    alt chat_state is None
        Handler->>RAG: realize_llm()
        RAG-->>Handler: llm instance
        Handler->>State: create ChatState
    end
    Handler->>State: chat_state.chat()
    State-->>Handler: response
    alt streaming
        Handler-->>Client: SSE stream
    else
        Handler-->>Client: JSON response
    end
```

## Class Hierarchy and Relationships

```mermaid
classDiagram
    class RAGWorkflow {
        +config: GlobalConfig
        +storage_strategy: StorageStrategy
        +embedding: BaseEmbedding
        +llm: LLM
        +initialize()
        +index_files()
        +retrieve_nodes()
        +realize_llm()
        +realize_embedding()
    }
    
    class QueryState {
        +config: QueryConfig
        +llm: LLM
        +retriever: BaseRetriever
        +query_engine: BaseQueryEngine
        +query()
    }
    
    class ChatState {
        +config: ChatConfig
        +llm: LLM
        +retriever: BaseRetriever
        +chat_engine: BaseChatEngine
        +memory: ChatMemoryBuffer
        +chat()
    }
    
    class GlobalConfig {
        +storage_strategy: str
        +retrieval: RetrievalConfig
        +query: QueryConfig
        +chat: ChatConfig
    }
    
    class BaseEmbedding {
        <<interface>>
        +embed_query()
        +embed_documents()
    }
    
    class LLM {
        <<interface>>
        +complete()
        +chat()
        +stream_complete()
        +stream_chat()
    }
    
    RAGWorkflow --> GlobalConfig
    RAGWorkflow --> BaseEmbedding
    RAGWorkflow --> LLM
    RAGWorkflow ..> QueryState : creates
    RAGWorkflow ..> ChatState : creates
    QueryState --> LLM
    ChatState --> LLM
    GlobalConfig --> QueryConfig
    GlobalConfig --> ChatConfig
```

## Provider Initialization Pattern

```mermaid
flowchart TD
    A[Config Object] --> B{Match Config Type}
    B -->|HuggingFaceEmbeddingConfig| C[Create HuggingFaceEmbedding]
    B -->|OllamaEmbeddingConfig| D[Create OllamaEmbedding]
    B -->|OpenAIEmbeddingConfig| E[Create OpenAIEmbedding]
    B -->|LiteLLMEmbeddingConfig| F[Create LiteLLMEmbedding]
    B -->|OllamaConfig| G[Create Ollama LLM]
    B -->|OpenAIConfig| H[Create OpenAI LLM]
    B -->|PerplexityConfig| I[Create Perplexity LLM]
    
    C & D & E & F --> J[BaseEmbedding Instance]
    G & H & I --> K[LLM Instance]
    
    J --> L[Return to RAGWorkflow]
    K --> L
```

## Error Handling Flow

```mermaid
flowchart TD
    A[User Action] --> B{Command Type}
    B -->|CLI| C[Try Command Execution]
    B -->|API| D[Try API Handler]
    
    C --> E{Success?}
    E -->|Yes| F[Return Result]
    E -->|No| G[error() function]
    G --> H[Print to stderr]
    H --> I[sys.exit(1)]
    
    D --> J{Success?}
    J -->|Yes| K[Return Response]
    J -->|No| L[HTTPException]
    L --> M[Status Code + Detail]
    M --> N[JSON Error Response]
```

## Performance Optimization Points

```mermaid
graph TD
    subgraph "Caching Layer"
        HC[Hash Cache<br/>Deduplication]
        EC[Embedding Cache<br/>Reuse computed]
        RC[Response Cache<br/>Query results]
    end
    
    subgraph "Parallel Processing"
        MW[Multi-Worker<br/>Document ingestion]
        BP[Batch Processing<br/>Embeddings]
        AP[Async Processing<br/>API requests]
    end
    
    subgraph "Storage Optimization"
        IX[Index Selection<br/>IVFFlat vs HNSW]
        CP[Connection Pooling<br/>PostgreSQL]
        LP[Lazy Loading<br/>On-demand imports]
    end
    
    HC --> |Reduces| DUP[Duplicate Processing]
    EC --> |Saves| COMP[Computation Time]
    RC --> |Improves| LAT[Response Latency]
    MW --> |Increases| THR[Throughput]
    BP --> |Optimizes| GPU[GPU Utilization]
    AP --> |Handles| CONC[Concurrent Requests]
    IX --> |Speeds| SIM[Similarity Search]
    CP --> |Reduces| CONN[Connection Overhead]
    LP --> |Reduces| MEM[Memory Footprint]
```

## Refactoring Opportunities

Based on the architecture analysis, key refactoring opportunities:

1. **Extract Provider Factory**: Centralize provider initialization logic
2. **Split RAGWorkflow**: Separate concerns into focused classes
3. **Abstract Configuration**: Create base classes for common config patterns
4. **Standardize Error Handling**: Implement consistent exception hierarchy
5. **Modularize Storage**: Create storage adapter interface
6. **Optimize Imports**: Implement lazy loading for heavy dependencies
7. **Simplify State Management**: Extract state management to dedicated classes
8. **Create Plugin System**: Enable dynamic provider registration

## Deployment Architecture

```mermaid
graph TB
    subgraph "Production Deployment"
        LB[Load Balancer]
        API1[API Server 1]
        API2[API Server 2]
        APIn[API Server N]
        
        PG_MAIN[(PostgreSQL<br/>Primary)]
        PG_READ[(PostgreSQL<br/>Read Replicas)]
        
        REDIS[Redis Cache]
        
        LB --> API1 & API2 & APIn
        API1 & API2 & APIn --> PG_MAIN
        API1 & API2 & APIn --> PG_READ
        API1 & API2 & APIn --> REDIS
    end
    
    subgraph "Development"
        DEV[Single Instance]
        CACHE[File Cache]
        DEV --> CACHE
    end
```

This architectural documentation provides a comprehensive view of the rag-client system, showing all major components, their interactions, data flows, and deployment patterns.