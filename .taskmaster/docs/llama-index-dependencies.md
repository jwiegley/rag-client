# llama-index Dependencies and Usage Patterns

## Overview
The rag-client project extensively uses llama-index as its core RAG framework with 36 unique modules imported and 21 package dependencies.

## Package Dependencies from requirements.txt

### Core Package
1. **llama-index-core** (>=0.12.34.post1) - Core framework functionality

### Embedding Packages (5 packages)
2. **llama-index-embeddings-huggingface** (>=0.5.3)
3. **llama-index-embeddings-ollama** (>=0.6.0)
4. **llama-index-embeddings-openai** (>=0.3.1)
5. **llama-index-embeddings-openai-like** (>=0.1.0)
6. **llama-index-embeddings-litellm** (>=0.3.0)

### LLM Packages (9 packages)
7. **llama-index-llms-llama-cpp** (>=0.4.0)
8. **llama-index-llms-ollama** (>=0.5.4)
9. **llama-index-llms-openai** (>=0.3.38)
10. **llama-index-llms-openai-like** (>=0.3.4)
11. **llama-index-llms-litellm** (>=0.5.1)
12. **llama-index-llms-perplexity** (>=0.3.3)
13. **llama-index-llms-lmstudio** (>=0.3.0)
14. **llama-index-llms-openrouter** (>=0.3.1)
15. **llama-index-llms-mlx** (>=0.3.0)

### Storage Packages (3 packages)
16. **llama-index-vector-stores-postgres** (>=0.5.1)
17. **llama-index-storage-docstore-postgres** (>=0.3.0)
18. **llama-index-storage-index-store-postgres** (>=0.4.0)

### Reader Packages (1 package)
19. **llama-index-readers-file** (>=0.4.7)

### Parsing Package (1 package)
20. **llama-parse** (>=0.6.20)

### Support Package (1 package)
21. **llama-cpp-python** (>=0.3.9) - C++ bindings for LLama models

## Module Import Analysis

### Core Framework Imports (11 modules)

#### Main Core Module (rag.py lines 38-48)
```python
from llama_index.core import (
    Document,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    SummaryIndex,
    TreeIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
```

#### Constants (rag.py lines 49-55)
```python
from llama_index.core.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
```

### Schema and Response Types (4 modules)

#### Response Schema (rag.py lines 56-58, main.py lines 16-19)
```python
from llama_index.core.query_engine.custom import STR_OR_RESPONSE_TYPE
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.base.response.schema import (
    StreamingResponse,
    Response,
)
```

#### Core Schema (rag.py lines 114-121)
```python
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageNode,
    MetadataMode,
    NodeRelationship,
    TextNode,
)
```

### Chat Engine Components (5 modules)

#### Chat Engine Types (rag.py lines 59-68, chat.py lines 4-8)
```python
from llama_index.core.chat_engine import (
    ChatMode,
    CondenseQuestionChatEngine,
)
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
    ChatResponseMode,
    StreamingChatResponse,
)
```

### Embeddings (2 modules)

#### Base Embeddings (rag.py lines 57, 70)
```python
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.embeddings import BaseEmbedding
```

### Evaluation Components (2 modules)

#### Evaluation Metrics (rag.py lines 72-77)
```python
from llama_index.core.evaluation import (
    BaseEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    GuidelineEvaluator,
)
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
```

### Extractors (1 module)

#### Metadata Extractors (rag.py lines 78-84)
```python
from llama_index.core.extractors import (
    BaseExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
```

### Index Components (3 modules)

#### Index Types (rag.py lines 85-87)
```python
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.keyword_table.base import BaseKeywordTableIndex
from llama_index.core.ingestion import IngestionPipeline
```

### LLM Components (2 modules)

#### LLM Interfaces (rag.py lines 88-89, api.py line 13)
```python
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
```

### Memory Components (2 modules)

#### Memory Buffers (rag.py line 90, chat.py line 16)
```python
from llama_index.core.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
```

### Node Parser Components (1 module)

#### Node Parsers (rag.py lines 91-97)
```python
from llama_index.core.node_parser import (
    CodeSplitter,
    JSONNodeParser,
    NodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
)
```

### Query Engine Components (1 module)

#### Query Engines (rag.py lines 98-106)
```python
from llama_index.core.query_engine import (
    BaseQueryEngine,
    CitationQueryEngine,
    CustomQueryEngine,
    MultiStepQueryEngine,
    RetrieverQueryEngine,
    RetryQueryEngine,
    RetrySourceQueryEngine,
)
```

### Response Synthesizers (2 modules)

#### Response Synthesis (rag.py lines 108-111, chat.py lines 18-22)
```python
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    ResponseMode,
)
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
```

### Retriever Components (3 modules)

#### Retrievers (rag.py lines 112-113, chat.py line 2)
```python
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.base.base_retriever import BaseRetriever
```

### Storage Components (6 modules)

#### Storage Context (rag.py lines 122-126, main.py line 20)
```python
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.storage_context import DEFAULT_PERSIST_DIR
from llama_index.core.vector_stores.simple import SimpleVectorStore
```

### Settings and Configuration (1 module)

#### Settings (chat.py line 24)
```python
from llama_index.core.settings import Settings
```

### Other Components (4 modules)

#### Callbacks (chat.py line 3)
```python
from llama_index.core.callbacks import CallbackManager
```

#### Postprocessors (chat.py line 17)
```python
from llama_index.core.postprocessor.types import BaseNodePostprocessor
```

#### Tools (chat.py line 25)
```python
from llama_index.core.tools import ToolOutput
```

#### Data Structures (rag.py line 69)
```python
from llama_index.core.data_structs.data_structs import IndexDict
```

#### Readers (rag.py line 107)
```python
from llama_index.core.readers.base import BaseReader
```

## Provider-Specific Implementations

### Embedding Providers (7 modules)
```python
# HuggingFace
from llama_index.embeddings.huggingface.base import DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# OpenAI
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
)

# OpenAI-like
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

# LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding
```

### LLM Providers (14 modules)
```python
# LlamaCPP
import llama_index.llms.llama_cpp.base
from llama_index.llms.llama_cpp import LlamaCPP

# Ollama
import llama_index.llms.ollama.base
from llama_index.llms.ollama import Ollama

# OpenAI
from llama_index.llms.openai.base import DEFAULT_OPENAI_MODEL
from llama_index.llms.openai import OpenAI

# OpenAI-like
from llama_index.llms.openai_like import OpenAILike

# LiteLLM
from llama_index.llms.litellm import LiteLLM

# OpenRouter
import llama_index.llms.openrouter.base
from llama_index.llms.openrouter import OpenRouter

# Perplexity
from llama_index.llms.perplexity import Perplexity

# LMStudio
import llama_index.llms.lmstudio.base
from llama_index.llms.lmstudio import LMStudio

# MLX
import llama_index.llms.mlx.base
from llama_index.llms.mlx import MLXLLM
```

### Storage Providers (3 modules)
```python
# PostgreSQL Storage
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.index_store.postgres import PostgresIndexStore
from llama_index.vector_stores.postgres import PGVectorStore
```

## Usage Patterns by Feature

### 1. Document Ingestion
- **SimpleDirectoryReader**: File loading and parsing
- **Document**: Document creation and management
- **IngestionPipeline**: Processing pipeline

### 2. Text Processing
- **SentenceSplitter**: Basic text chunking
- **SemanticSplitterNodeParser**: Semantic-aware chunking
- **CodeSplitter**: Code-aware splitting
- **JSONNodeParser**: JSON document parsing
- **SentenceWindowNodeParser**: Context window splitting

### 3. Embedding Generation
- Multiple provider implementations (HuggingFace, OpenAI, Ollama, etc.)
- **BaseEmbedding**: Common interface
- Batch processing with DEFAULT_EMBED_BATCH_SIZE

### 4. Vector Storage
- **SimpleVectorStore**: In-memory storage
- **PGVectorStore**: PostgreSQL with pgvector
- **StorageContext**: Storage abstraction layer

### 5. Indexing
- **VectorStoreIndex**: Primary vector index
- **SummaryIndex**: Summary-based index
- **TreeIndex**: Hierarchical index
- **BaseKeywordTableIndex**: Keyword-based index

### 6. Retrieval
- **BaseRetriever**: Common retriever interface
- **QueryFusionRetriever**: Multi-query fusion
- FUSION_MODES for retrieval strategies

### 7. Query Processing
- **RetrieverQueryEngine**: Standard query engine
- **CitationQueryEngine**: With source citations
- **MultiStepQueryEngine**: Multi-step reasoning
- **RetryQueryEngine**: With retry logic
- **RetrySourceQueryEngine**: Source-aware retry

### 8. Response Generation
- **get_response_synthesizer**: Response synthesis factory
- **ResponseMode**: Response generation strategies
- **BaseSynthesizer**: Custom synthesis

### 9. Chat Functionality
- **CondenseQuestionChatEngine**: Question condensation
- **ContextChatEngine**: Context-aware chat
- **ChatMemoryBuffer**: Conversation memory
- **ChatSummaryMemoryBuffer**: Summarized memory

### 10. Evaluation
- **CorrectnessEvaluator**: Answer correctness
- **FaithfulnessEvaluator**: Faithfulness to source
- **GuidelineEvaluator**: Custom guidelines

### 11. Metadata Extraction
- **KeywordExtractor**: Keyword extraction
- **SummaryExtractor**: Summary generation
- **TitleExtractor**: Title extraction
- **QuestionsAnsweredExtractor**: Question extraction

## Feature-to-Dependency Matrix

| Feature | Required Packages |
|---------|------------------|
| Basic RAG | llama-index-core |
| HuggingFace Embeddings | llama-index-embeddings-huggingface |
| OpenAI Integration | llama-index-llms-openai, llama-index-embeddings-openai |
| Ollama Integration | llama-index-llms-ollama, llama-index-embeddings-ollama |
| Local Models | llama-index-llms-llama-cpp, llama-cpp-python |
| PostgreSQL Storage | llama-index-vector-stores-postgres, llama-index-storage-docstore-postgres, psycopg2-binary |
| File Reading | llama-index-readers-file |
| Advanced Parsing | llama-parse |
| Multiple LLM Providers | llama-index-llms-* packages |

## Version Compatibility Notes

1. **Core Version**: Uses 0.12.34.post1 or higher - relatively recent version
2. **Provider Packages**: All use compatible versions with core
3. **No Version Conflicts**: All llama-index packages aligned
4. **Python Compatibility**: Requires Python 3.8+ based on llama-index requirements

## Observations and Recommendations

### Current State
1. **Heavy Framework Dependency**: 36 unique llama-index modules imported
2. **Provider Proliferation**: 9 LLM providers, 5 embedding providers
3. **Monolithic Imports**: Most imports concentrated in rag.py
4. **Full Feature Usage**: Using nearly all llama-index features

### Potential Issues
1. **Large Dependency Footprint**: 21 llama-index packages increases attack surface
2. **Provider Redundancy**: Multiple providers for same functionality
3. **Import Complexity**: Deep module paths make refactoring difficult
4. **Version Lock-in**: Heavy coupling to llama-index API

### Refactoring Recommendations
1. **Lazy Loading**: Load providers only when needed
2. **Provider Abstraction**: Create unified provider interface
3. **Feature Flags**: Enable/disable features to reduce dependencies
4. **Import Organization**: Group imports by feature area
5. **Dependency Injection**: Reduce direct llama-index coupling