"""Core RAG workflow implementation."""

import base64
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union, cast

from llama_index.core import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
    VectorStoreIndex,
    load_indices_from_storage,
)
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import (
    BaseExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.keyword_table.base import BaseKeywordTableIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import (
    CodeSplitter,
    JSONNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import BaseNode, Document, TransformComponent
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

# Provider imports moved to factory.py
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.index_store.postgres import PostgresIndexStore
from llama_index.vector_stores.postgres import PGVectorStore

from ..config.models import (
    CodeSplitterConfig,
    Config,
    EmbeddingConfig,
    ExtractorConfig,
    JSONNodeParserConfig,
    KeywordExtractorConfig,
    LLMConfig,
    PostgresVectorStoreConfig,
    QuestionsAnsweredExtractorConfig,
    RetrievalConfig,
    SemanticSplitterConfig,
    SentenceSplitterConfig,
    SentenceWindowSplitterConfig,
    SplitterConfig,
    SummaryExtractorConfig,
    TitleExtractorConfig,
)
from ..providers.factory import create_embedding_provider, create_llm_provider
from ..storage.postgres import PostgresDetails
from ..types import (
    ConfigDict,
    DocumentList,
    EmbeddingProvider,
    LLMProvider,
    NodeList,
    Retriever,
    ScoredNodeList,
    StorageBackend,
)
from ..utils.helpers import cache_dir, collection_hash, error
from ..utils.readers import MailParser, OrgReader
from .retrieval import CustomRetriever

DEFAULT_PERSIST_DIR = "./storage"


@dataclass
class RAGWorkflow:
    """Main RAG workflow orchestrator.
    
    The RAGWorkflow class manages the complete Retrieval-Augmented Generation pipeline,
    including document ingestion, embedding generation, index creation, and retrieval.
    It supports multiple storage backends (ephemeral and persistent), various embedding
    and LLM providers, and flexible configuration options.
    
    Attributes:
        logger: Logger instance for workflow logging and debugging.
        config: Configuration object containing all workflow settings including
            embedding models, LLMs, storage backends, and processing parameters.
    
    Example:
        >>> from pathlib import Path
        >>> config = RAGWorkflow.load_config(Path("config.yaml"))
        >>> workflow = RAGWorkflow(logger=get_logger(), config=config)
        >>> retriever = workflow.load_retriever(
        ...     input_files=[Path("docs/")],
        ...     verbose=True
        ... )
        >>> results = workflow.retrieve_nodes(retriever, "What is RAG?")
    
    Note:
        The workflow supports both cached and fresh indexing. Use index_files=True
        to force re-indexing of documents even if a cache exists.
    """
    logger: logging.Logger
    config: Config

    @classmethod
    def load_config(cls, path: Path) -> Config:
        """Load configuration from YAML file.
        
        Parses a YAML configuration file and creates a Config object with all
        necessary settings for the RAG workflow. The configuration file should
        define embedding models, LLMs, storage backends, and processing parameters.
        
        Args:
            path: Path to the YAML configuration file. Must be a valid YAML
                file containing a Config object definition.
            
        Returns:
            Config object with all workflow settings loaded and validated.
            
        Raises:
            SystemExit: If the config file doesn't exist, is invalid YAML,
                or doesn't define a proper Config object.
            
        Example:
            >>> config = RAGWorkflow.load_config(Path("config.yaml"))
            >>> workflow = RAGWorkflow(logger=get_logger(), config=config)
        
        Config File Example:
            ```yaml
            retrieval:
              embedding:
                type: huggingface
                model: BAAI/bge-small-en-v1.5
              splitter:
                type: sentence
                chunk_size: 512
                chunk_overlap: 50
            query:
              llm:
                type: ollama
                model: llama2
            ```
        """
        if os.path.isfile(path):
            cfg = Config.from_yaml_file(  # pyright: ignore[reportUnknownMemberType]
                str(path)
            )
            if isinstance(cfg, Config):
                return cfg
            else:
                error("Config file should define a single Config object")
        else:
            error(f"Cannot read config file {path}")

    @classmethod
    def __postgres_stores(
        cls,
        config: PostgresVectorStoreConfig,
        embedding_dimensions: int,
    ) -> Tuple[PostgresDocumentStore, PostgresIndexStore, PGVectorStore]:
        """Initialize PostgreSQL storage components.
        
        Args:
            config: PostgreSQL configuration
            embedding_dimensions: Dimensionality of embeddings
            
        Returns:
            Tuple of (document store, index store, vector store)
        """
        docstore: PostgresDocumentStore = PostgresDocumentStore.from_uri(
            uri=config.connection,
            table_name="docstore",
        )
        index_store: PostgresIndexStore = PostgresIndexStore.from_uri(
            uri=config.connection,
            table_name="indexstore",
        )

        details: PostgresDetails = PostgresDetails(config.connection)

        vector_store: PGVectorStore = PGVectorStore.from_params(
            connection_string=config.connection,
            database=details.database,
            host=details.host,
            password=details.password,
            port=str(details.port),
            user=details.user,
            table_name="vectorstore",
            embed_dim=embedding_dimensions,
            hybrid_search=config.hybrid_search,
            hnsw_kwargs={
                "hnsw_m": config.hnsw_m,
                "hnsw_ef_construction": config.hnsw_ef_construction,
                "hnsw_ef_search": config.hnsw_ef_search,
                "hnsw_dist_method": config.hnsw_dist_method,
            },
        )
        return docstore, index_store, vector_store

    # Note: __load_component is now deprecated in favor of factory pattern
    # but kept for backward compatibility

    @classmethod
    def __load_embedding(
        cls,
        config: EmbeddingConfig,
        verbose: bool = False,
    ) -> BaseEmbedding:
        """Load an embedding model.
        
        Args:
            config: Embedding configuration
            verbose: Whether to show progress
            
        Returns:
            Initialized embedding model
            
        Raises:
            SystemExit: If embedding fails to load
        """
        try:
            return create_embedding_provider(config, verbose)
        except Exception as e:
            error(f"Failed to load embedding: {e}")

    @classmethod
    def __load_llm(
        cls,
        config: LLMConfig,
        verbose: bool = False,
    ) -> LLM:
        """Load an LLM model.
        
        Args:
            config: LLM configuration
            verbose: Whether to show progress
            
        Returns:
            Initialized LLM model
            
        Raises:
            SystemExit: If LLM fails to load
        """
        try:
            return create_llm_provider(config, verbose)
        except Exception as e:
            error(f"Failed to load LLM: {e}")

    @classmethod
    def realize_llm(
        cls,
        config: Optional[LLMConfig],
        verbose: bool = False,
    ) -> Union[LLM, NoReturn]:
        """Realize an LLM from configuration.
        
        Factory method that creates and initializes an LLM instance based on
        the provided configuration. Supports multiple providers including
        OpenAI, Ollama, LiteLLM, Perplexity, and others.
        
        Args:
            config: Optional LLM configuration specifying provider and model.
                If None, the method will exit with an error.
            verbose: If True, shows initialization progress and debug info.
            
        Returns:
            Initialized LLM instance ready for text generation.
            
        Raises:
            SystemExit: If config is None or LLM initialization fails.
            
        Example:
            >>> from rag_client.config.models import OllamaConfig
            >>> config = OllamaConfig(model="llama2", base_url="http://localhost:11434")
            >>> llm = RAGWorkflow.realize_llm(config, verbose=True)
            >>> response = llm.complete("What is machine learning?")
        
        Supported Providers:
            - OpenAI (GPT-3.5, GPT-4)
            - Ollama (local models)
            - LiteLLM (multi-provider proxy)
            - Perplexity (online models)
            - LMStudio (local models)
            - OpenRouter (multi-provider)
            - MLX (Apple Silicon optimized)
        """
        llm = (
            cls.__load_llm(
                config=config,
                verbose=verbose,
            )
            if config is not None
            else None
        )
        if llm is None:
            error(f"Failed to start LLM: {config}")
        else:
            return llm

    @classmethod
    def __determine_fingerprint(
        cls,
        input_files: List[Path],
        config: RetrievalConfig,
    ) -> str:
        """Generate fingerprint for cache identification.
        
        Args:
            input_files: List of input file paths
            config: Retrieval configuration
            
        Returns:
            Base64-encoded fingerprint string (32 chars)
        """
        # jww (2025-06-25): Include configuration details that might affect
        # the persisted vector index.
        fingerprint = [collection_hash(input_files), repr(config)]
        final_hash = "\n".join(fingerprint).encode("utf-8")
        final_base64 = base64.b64encode(final_hash).decode("utf-8")
        return final_base64[0:32]

    @classmethod
    def __read_documents(
        cls,
        input_files: List[Path],
        num_workers: Optional[int] = None,
        recursive: bool = False,
        verbose: bool = False,
    ) -> Iterable[Document]:
        """Read documents from input files.
        
        Args:
            input_files: List of file paths to read
            num_workers: Number of workers for parallel processing
            recursive: Whether to recursively read directories
            verbose: Whether to show progress
            
        Returns:
            Iterable of loaded Document objects
        """
        file_extractor: dict[str, BaseReader] = {
            ".org": OrgReader(),
            ".eml": MailParser(),
        }
        return SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=file_extractor,
            recursive=recursive,
        ).load_data(num_workers=num_workers, show_progress=verbose)

    @classmethod
    def __load_splitter(
        cls,
        splitter: SplitterConfig,
        verbose: bool = False,
    ) -> Union[TransformComponent, NoReturn]:
        """Load a text splitter component.
        
        Args:
            splitter: Splitter configuration
            verbose: Whether to show progress
            
        Returns:
            Initialized splitter component
            
        Raises:
            SystemExit: If splitter requires missing embedding model
        """
        match splitter:
            case SentenceSplitterConfig():
                return SentenceSplitter(
                    chunk_size=splitter.chunk_size,
                    chunk_overlap=splitter.chunk_overlap,
                    include_metadata=splitter.include_metadata,
                )
            case SentenceWindowSplitterConfig():
                return SentenceWindowNodeParser.from_defaults(
                    window_size=splitter.window_size,
                    window_metadata_key=splitter.window_metadata_key,
                    original_text_metadata_key=splitter.original_text_metadata_key,
                )
            case SemanticSplitterConfig():
                if splitter.embedding is not None:
                    embed_llm = RAGWorkflow.__load_embedding(
                        splitter.embedding,
                        verbose=verbose,
                    )
                else:
                    error("Semantic splitter needs an embedding model")

                return SemanticSplitterNodeParser(
                    buffer_size=splitter.buffer_size,
                    breakpoint_percentile_threshold=splitter.breakpoint_percentile_threshold,
                    embed_model=embed_llm,
                    include_metadata=splitter.include_metadata,
                )
            case JSONNodeParserConfig():
                return JSONNodeParser(
                    include_metadata=splitter.include_metadata,
                    include_prev_next_rel=splitter.include_prev_next_rel,
                )
            case CodeSplitterConfig():
                return CodeSplitter(
                    language=splitter.language,
                    chunk_lines=splitter.chunk_lines,
                    chunk_lines_overlap=splitter.chunk_lines_overlap,
                    max_chars=splitter.max_chars,
                )

    @classmethod
    def __load_extractor(
        cls,
        config: ExtractorConfig,
        verbose: bool = False,
    ) -> BaseExtractor:
        """Load a metadata extractor.
        
        Args:
            config: Extractor configuration
            verbose: Whether to show progress
            
        Returns:
            Initialized extractor
        """
        match config:
            case KeywordExtractorConfig():
                llm = RAGWorkflow.realize_llm(config.llm, verbose=verbose)
                return KeywordExtractor(
                    keywords=config.keywords,
                    llm=llm,
                    show_progress=verbose,
                )
            case SummaryExtractorConfig():
                llm = RAGWorkflow.realize_llm(config.llm, verbose=verbose)
                return SummaryExtractor(
                    summaries=config.summaries or [],
                    llm=llm,
                    show_progress=verbose,
                )
            case TitleExtractorConfig():
                llm = RAGWorkflow.realize_llm(config.llm, verbose=verbose)
                return TitleExtractor(
                    nodes=config.nodes,
                    llm=llm,
                    show_progress=verbose,
                )
            case QuestionsAnsweredExtractorConfig():
                llm = RAGWorkflow.realize_llm(config.llm, verbose=verbose)
                return QuestionsAnsweredExtractor(
                    questions=config.questions,
                    llm=llm,
                    show_progress=verbose,
                )

    def __process_documents(
        self,
        documents: Iterable[Document],
        embed_llm: BaseEmbedding,
        num_workers: Optional[int] = None,
        verbose: bool = False,
    ) -> Sequence[BaseNode]:
        """Process documents through ingestion pipeline.
        
        Args:
            documents: Input documents to process
            embed_llm: Embedding model for vectorization
            num_workers: Number of workers for parallel processing
            verbose: Whether to show progress
            
        Returns:
            Sequence of processed nodes
        """
        transformations: list[TransformComponent] = [
            self.__load_splitter(
                splitter=self.config.retrieval.splitter or SentenceSplitterConfig(),
                verbose=verbose,
            )
        ]

        transformations.extend(
            [
                self.__load_extractor(
                    entry,
                    verbose=verbose,
                )
                for entry in self.config.retrieval.extractors or []
            ]
        )

        transformations.append(embed_llm)

        # ingest_cache = IngestionCache(
        #     cache=RedisCache.from_host_and_port(host="127.0.0.1", port=6379),
        #     collection="my_test_cache",
        # )

        pipeline: IngestionPipeline = IngestionPipeline(
            transformations=transformations,
            # cache=ingest_cache,
            docstore=SimpleDocumentStore(),
        )
        return pipeline.run(
            documents=documents,
            num_workers=num_workers,
            show_progress=verbose,
        )

    def __build_vector_index(
        self,
        embed_llm: BaseEmbedding,
        nodes: Sequence[BaseNode],
        storage_context: StorageContext,
        verbose: bool = False,
    ) -> Tuple[VectorStoreIndex, Optional[BaseKeywordTableIndex]]:
        """Build vector and keyword indices from nodes.
        
        Args:
            embed_llm: Embedding model for vectorization
            nodes: Processed nodes to index
            storage_context: Storage context for persistence
            verbose: Whether to show progress
            
        Returns:
            Tuple of (vector index, optional keyword index)
        """
        index_structs = storage_context.index_store.index_structs()
        for struct in index_structs:
            storage_context.index_store.delete_index_struct(key=struct.index_id)

        vector_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_llm,
            show_progress=verbose,
        )

        if (
            self.config.retrieval.keywords is not None
            and self.config.retrieval.keywords.collect
        ):
            if self.config.retrieval.keywords.llm is None:
                keyword_index = SimpleKeywordTableIndex(
                    nodes,
                    storage_context=storage_context,
                    show_progress=verbose,
                )
            else:
                llm = self.realize_llm(self.config.retrieval.keywords.llm)
                keyword_index = KeywordTableIndex(
                    nodes,
                    storage_context=storage_context,
                    show_progress=verbose,
                    llm=llm,
                )
        else:
            keyword_index = None

        return vector_index, keyword_index

    def __persist_dir(self, input_files: List[Path]) -> Path:
        """Get persistence directory for cache.
        
        Args:
            input_files: Input file paths
            
        Returns:
            Path to persistence directory
        """
        return cache_dir() / self.__determine_fingerprint(
            input_files, self.config.retrieval
        )

    def __load_storage_context(
        self,
        persist_dir: Optional[Path] = None,
    ) -> StorageContext:
        """Load or create storage context.
        
        Args:
            persist_dir: Optional persistence directory
            
        Returns:
            Initialized storage context
        """
        if (
            self.config.retrieval.vector_store is not None
            and isinstance(
                self.config.retrieval.vector_store, PostgresVectorStoreConfig
            )
            and self.config.retrieval.embedding is not None
        ):
            docstore, index_store, vector_store = self.__postgres_stores(
                config=self.config.retrieval.vector_store,
                embedding_dimensions=self.config.retrieval.vector_store.dimensions,
            )
        elif persist_dir is not None and os.path.isdir(persist_dir):
            docstore = SimpleDocumentStore.from_persist_dir(str(persist_dir))
            index_store = SimpleIndexStore.from_persist_dir(str(persist_dir))
            vector_store = SimpleVectorStore.from_persist_dir(str(persist_dir))
        else:
            docstore = SimpleDocumentStore()
            index_store = SimpleIndexStore()
            vector_store = SimpleVectorStore()

        return StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store,
            vector_store=vector_store,
            persist_dir=(
                str(persist_dir) if persist_dir is not None else DEFAULT_PERSIST_DIR
            ),
        )

    def __ingest_documents(
        self,
        embed_llm: BaseEmbedding,
        storage_context: StorageContext,
        input_files: List[Path],
        num_workers: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[VectorStoreIndex, Optional[BaseKeywordTableIndex]]:
        """Ingest documents and build indices.
        
        Args:
            embed_llm: Embedding model
            storage_context: Storage context
            input_files: Files to ingest
            num_workers: Number of workers
            verbose: Whether to show progress
            
        Returns:
            Tuple of (vector index, optional keyword index)
        """
        documents = self.__read_documents(
            input_files=input_files,
            num_workers=num_workers,
            verbose=verbose,
        )

        nodes = self.__process_documents(
            documents=documents,
            embed_llm=embed_llm,
            num_workers=num_workers,
            verbose=verbose,
        )

        vector_index, keyword_index = self.__build_vector_index(
            embed_llm=embed_llm,
            nodes=nodes,
            storage_context=storage_context,
            verbose=verbose,
        )

        vector_index.set_index_id("vector_index")
        if keyword_index is not None:
            keyword_index.set_index_id("keyword_index")

        return vector_index, keyword_index

    def __save_indices(
        self,
        storage_context: StorageContext,
        persist_dir: Optional[Path],
    ) -> None:
        """Save indices to persistence.
        
        Args:
            storage_context: Storage context to persist
            persist_dir: Directory to persist to
        """
        if self.config.retrieval.vector_store is not None:
            pass
        elif persist_dir is not None:
            storage_context.persist(  # pyright: ignore[reportUnknownMemberType]
                persist_dir=str(persist_dir)
            )

    def __load_indices(
        self,
        embed_llm: BaseEmbedding,
        storage_context: StorageContext,
    ) -> Tuple[Optional[VectorStoreIndex], Optional[BaseKeywordTableIndex]]:
        """Load indices from storage.
        
        Args:
            embed_llm: Embedding model
            storage_context: Storage context to load from
            
        Returns:
            Tuple of (optional vector index, optional keyword index)
        """
        keyword_index = None

        indices: list[BaseIndex[IndexDict]]
        indices = load_indices_from_storage(  # pyright: ignore[reportUnknownVariableType]
            storage_context=storage_context,
            embed_model=embed_llm,
            # This doesn't actually need an llm, but it tries to
            # load one if it's None
            llm=embed_llm,
            index_ids=(
                ["vector_index", "keyword_index"]
                if self.config.retrieval.keywords is not None
                and self.config.retrieval.keywords.collect
                else ["vector_index"]
            ),
        )
        if (
            self.config.retrieval.keywords is not None
            and self.config.retrieval.keywords.collect
        ):
            [vi, ki] = indices
            vector_index = cast(VectorStoreIndex, vi)
            keyword_index = cast(BaseKeywordTableIndex, ki)
        else:
            [vi] = indices
            vector_index = cast(VectorStoreIndex, vi)

        return vector_index, keyword_index

    def __load_indices_from_disk(
        self,
        storage_context: StorageContext,
        input_files: Optional[List[Path]],
        embed_llm: BaseEmbedding,
        persist_dir: Optional[Path] = None,
        num_workers: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[Optional[VectorStoreIndex], Optional[BaseKeywordTableIndex]]:
        """Load indices from disk storage.
        
        Args:
            storage_context: Storage context
            input_files: Input files to index
            embed_llm: Embedding model
            persist_dir: Persistence directory
            num_workers: Number of workers
            verbose: Whether to show progress
            
        Returns:
            Tuple of (optional vector index, optional keyword index)
            
        Raises:
            SystemExit: If input_files is None
        """
        if input_files is None:
            error("Cannot create vector index without input files")

        vector_index, keyword_index = self.__ingest_documents(
            num_workers=num_workers,
            embed_llm=embed_llm,
            storage_context=storage_context,
            input_files=input_files,
            verbose=verbose,
        )

        self.__save_indices(
            storage_context,
            persist_dir=persist_dir,
        )

        return (vector_index, keyword_index)

    def __load_indices_from_cache(
        self,
        storage_context: StorageContext,
        embed_llm: BaseEmbedding,
    ) -> Tuple[Optional[VectorStoreIndex], Optional[BaseKeywordTableIndex]]:
        """Load indices from cache.
        
        Args:
            storage_context: Storage context
            embed_llm: Embedding model
            
        Returns:
            Tuple of (optional vector index, optional keyword index)
        """
        try:
            vector_index, keyword_index = self.__load_indices(
                embed_llm=embed_llm,
                storage_context=storage_context,
            )
        except ValueError:
            vector_index = None
            keyword_index = None

        return (vector_index, keyword_index)

    def __ingest_files(
        self,
        input_files: Optional[List[Path]],
        embed_llm: BaseEmbedding,
        index_files: bool = False,
        num_workers: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[Optional[VectorStoreIndex], Optional[BaseKeywordTableIndex]]:
        """Ingest files and create indices.
        
        Args:
            input_files: Files to ingest
            embed_llm: Embedding model
            index_files: Whether to force re-indexing
            num_workers: Number of workers
            verbose: Whether to show progress
            
        Returns:
            Tuple of (optional vector index, optional keyword index)
            
        Raises:
            SystemExit: If embedding model is missing
        """
        persist_dir: Path | None = None

        if input_files is None:
            pass
        elif self.config.retrieval.embedding is None:
            error("Cannot ingest files without an embedding model")
        else:
            persist_dir = self.__persist_dir(input_files)

        storage_context = self.__load_storage_context(
            persist_dir=persist_dir,
        )

        vector_index = None
        keyword_index = None

        if not index_files:
            (vector_index, keyword_index) = self.__load_indices_from_cache(
                storage_context,
                embed_llm,
            )

        if input_files is not None and vector_index is None and keyword_index is None:
            (vector_index, keyword_index) = self.__load_indices_from_disk(
                storage_context,
                input_files,
                embed_llm,
                persist_dir,
                num_workers,
                verbose,
            )

        return (vector_index, keyword_index)

    def __retriever_from_index(
        self,
        vector_index: Optional[VectorStoreIndex],
        keyword_index: Optional[BaseKeywordTableIndex],
        top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        verbose: bool = False,
    ) -> Optional[BaseRetriever]:
        """Create retriever from indices.
        
        Args:
            vector_index: Vector index for dense retrieval
            keyword_index: Keyword index for sparse retrieval
            top_k: Number of top results for dense retrieval
            sparse_top_k: Number of top results for sparse retrieval
            verbose: Whether to show progress
            
        Returns:
            Combined retriever or None if no indices
        """
        if vector_index is not None:
            if (
                self.config.retrieval.vector_store is not None
                and isinstance(
                    self.config.retrieval.vector_store, PostgresVectorStoreConfig
                )
                and self.config.retrieval.fusion is not None
            ):
                vector_retriever = vector_index.as_retriever(
                    vector_store_query_mode="default",
                    similarity_top_k=top_k,
                    verbose=verbose,
                )
                text_retriever = vector_index.as_retriever(
                    vector_store_query_mode="sparse",
                    similarity_top_k=sparse_top_k,
                    verbose=verbose,
                )
                llm = self.realize_llm(self.config.retrieval.fusion.llm)
                vector_retriever = QueryFusionRetriever(
                    [vector_retriever, text_retriever],
                    similarity_top_k=top_k or DEFAULT_SIMILARITY_TOP_K,
                    num_queries=self.config.retrieval.fusion.num_queries,
                    mode=self.config.retrieval.fusion.mode,
                    llm=llm,
                    verbose=verbose,
                )
            else:
                self.logger.info("Load retriever from vector index")
                vector_retriever = vector_index.as_retriever(
                    similarity_top_k=top_k,
                    verbose=verbose,
                )
        else:
            self.logger.info("There is no vector index")
            vector_retriever = None

        if keyword_index is not None:
            keyword_retriever = keyword_index.as_retriever(
                verbose=verbose,
            )
        else:
            keyword_retriever = None

        if vector_retriever is not None:
            if keyword_retriever is not None:
                retriever = CustomRetriever(
                    vector_retriever=vector_retriever,
                    keyword_retriever=keyword_retriever,
                    verbose=verbose,
                )
            else:
                retriever = vector_retriever
        else:
            retriever = keyword_retriever

        return retriever

    @classmethod
    def __merge_nodes(
        cls,
        storage_context: StorageContext,
        nodes: List[BaseNode],
    ) -> None:
        """Merge nodes from storage context.
        
        Args:
            storage_context: Storage context containing nodes
            nodes: List to append nodes to
        """
        d = storage_context.vector_store.to_dict()
        embedding_dict = d["embedding_dict"]
        items = storage_context.docstore.docs.items()
        for doc_id, node in items:
            node.embedding = embedding_dict[doc_id]
            nodes.append(node)

    def load_retriever(
        self,
        input_files: Optional[List[Path]],
        num_workers: Optional[int] = None,
        embed_individually: bool = False,
        index_files: bool = False,
        top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        verbose: bool = False,
    ) -> Optional[BaseRetriever]:
        """Load retriever for document search.
        
        Creates or loads a retriever that can search through indexed documents.
        Supports both vector-based (dense) and keyword-based (sparse) retrieval,
        with optional fusion for hybrid search. The method handles caching
        automatically based on input file fingerprints.
        
        Args:
            input_files: List of file paths to index. Can be files or directories.
                If None, attempts to load from existing storage (e.g., database).
            num_workers: Number of parallel workers for document processing.
                If None, uses sequential processing.
            embed_individually: If True, processes each file separately before
                merging indices. Useful for large document sets to avoid memory issues.
            index_files: If True, forces re-indexing even if cache exists.
                Use when documents have changed but fingerprint hasn't.
            top_k: Number of top results to retrieve for dense/vector search.
                Defaults to configuration or DEFAULT_SIMILARITY_TOP_K.
            sparse_top_k: Number of top results for sparse/keyword search.
                Only used with hybrid retrieval configurations.
            verbose: If True, shows progress bars and detailed logging.
            
        Returns:
            BaseRetriever configured for document search, or None if no
            retriever could be created (e.g., no input files and no existing index).
            
        Raises:
            SystemExit: If embedding model is not configured but required.
            
        Example:
            >>> # Basic usage with local files
            >>> retriever = workflow.load_retriever(
            ...     input_files=[Path("docs/")],
            ...     verbose=True
            ... )
            
            >>> # Force re-indexing
            >>> retriever = workflow.load_retriever(
            ...     input_files=[Path("data.pdf")],
            ...     index_files=True,
            ...     top_k=10
            ... )
            
            >>> # Load from existing database
            >>> retriever = workflow.load_retriever(
            ...     input_files=None,  # Load from PostgreSQL
            ...     top_k=5
            ... )
        
        Note:
            - Cache is stored in XDG_CACHE_HOME/rag-client/ by default
            - PostgreSQL storage requires proper connection configuration
            - Large document sets may benefit from embed_individually=True
        """
        if self.config.retrieval.embedding is not None:
            embed_llm = self.__load_embedding(
                config=self.config.retrieval.embedding,
                verbose=verbose,
            )
        else:
            error("File ingestion requires an embedding model")

        if input_files is not None and embed_individually:
            vi_nodes: list[BaseNode] = []
            ki_nodes: list[BaseNode] = []

            for input_file in input_files:
                vector_index, keyword_index = self.__ingest_files(
                    num_workers=num_workers,
                    input_files=[input_file],
                    embed_llm=embed_llm,
                    index_files=index_files,
                    verbose=verbose,
                )
                if vector_index is not None:
                    self.__merge_nodes(vector_index.storage_context, vi_nodes)
                if keyword_index is not None:
                    self.__merge_nodes(keyword_index.storage_context, ki_nodes)

            vector_index = VectorStoreIndex(
                nodes=vi_nodes,
                embed_model=embed_llm,
            )
            if len(ki_nodes) > 0:
                keyword_index = SimpleKeywordTableIndex(
                    nodes=ki_nodes,
                )
            else:
                keyword_index = None
        else:
            # If there are no input files mentioned, this can only be the
            # situation where we are expected to load a vector index directly
            # from a database, such as Postgres, assuming that a collection
            # had been embedded there earlier using the "index" command
            # against some set of files.
            vector_index, keyword_index = self.__ingest_files(
                num_workers=num_workers,
                input_files=input_files,
                embed_llm=embed_llm,
                index_files=index_files,
                verbose=verbose,
            )

        retriever = self.__retriever_from_index(
            vector_index,
            keyword_index,
            top_k=top_k,
            sparse_top_k=sparse_top_k,
            verbose=verbose,
        )

        return retriever

    def retrieve_nodes(
        self, retriever: BaseRetriever, text: str
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant nodes for a query.
        
        Searches the indexed documents for content relevant to the query text.
        Returns the most relevant document chunks along with their metadata.
        
        Args:
            retriever: The retriever instance to use for search. Must be
                initialized via load_retriever() first.
            text: The query text to search for. Can be a question, keywords,
                or any natural language query.
            
        Returns:
            List of dictionaries, each containing:
                - 'text': The content of the retrieved document chunk
                - 'metadata': Dictionary with source file, page numbers, etc.
            
        Example:
            >>> retriever = workflow.load_retriever(input_files=[Path("docs/")])
            >>> results = workflow.retrieve_nodes(
            ...     retriever,
            ...     "How does RAG improve LLM responses?"
            ... )
            >>> for node in results:
            ...     print(f"Source: {node['metadata'].get('file_name')}")
            ...     print(f"Content: {node['text'][:200]}...")
        
        Note:
            The number of results returned depends on the top_k parameter
            set during retriever initialization.
        """
        nodes = retriever.retrieve(text)
        return [
            {
                "text": node.text,
                "metadata": node.metadata,
            }
            for node in nodes
        ]