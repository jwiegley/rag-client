"""Async document processing for RAG client.

This module provides async/concurrent document processing capabilities
for improved performance when indexing large document collections.
"""

import asyncio
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from llama_index.core.schema import BaseNode, Document

from ..exceptions import DocumentProcessingError
from ..utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class AsyncDocumentProcessor:
    """Async document processor for parallel indexing.

    Provides concurrent document reading, parsing, and embedding generation
    for significantly faster indexing of large document collections.

    Features:
        - Parallel file reading with configurable worker count
        - Batched embedding generation
        - Progress tracking callbacks
        - Error handling with partial success support

    Example:
        >>> processor = AsyncDocumentProcessor(max_workers=4)
        >>> documents = await processor.read_documents_async(file_paths)
        >>> nodes = await processor.process_documents_async(documents, embed_model)
    """

    def __init__(
        self,
        max_workers: int | None = None,
        use_processes: bool = False,
        batch_size: int = 10,
    ):
        """Initialize async processor.

        Args:
            max_workers: Maximum concurrent workers (None = CPU count)
            use_processes: Use processes instead of threads for CPU-bound work
            batch_size: Batch size for embedding operations
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.batch_size = batch_size

        self._executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None

    def _get_executor(self) -> ThreadPoolExecutor | ProcessPoolExecutor:
        """Get or create executor."""
        if self._executor is None:
            if self.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    async def map_async(
        self,
        func: Callable[[T], Any],
        items: Iterable[T],
        desc: str = "Processing",
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[Any]:
        """Map a function over items concurrently.

        Args:
            func: Function to apply to each item
            items: Items to process
            desc: Description for logging
            on_progress: Optional callback(completed, total)

        Returns:
            List of results
        """
        items_list = list(items)
        total = len(items_list)
        completed = 0
        results = []
        errors = []

        loop = asyncio.get_event_loop()
        executor = self._get_executor()

        async def process_item(item: T) -> Any:
            nonlocal completed
            try:
                result = await loop.run_in_executor(executor, func, item)
                completed += 1
                if on_progress:
                    on_progress(completed, total)
                return result
            except Exception as e:
                errors.append((item, e))
                completed += 1
                if on_progress:
                    on_progress(completed, total)
                return None

        logger.info(f"{desc}: {total} items with {self.max_workers or 'auto'} workers")

        tasks = [process_item(item) for item in items_list]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        if errors:
            logger.warning(f"{len(errors)} items failed during {desc}")
            for item, error in errors[:5]:  # Log first 5 errors
                logger.debug(f"  {item}: {error}")

        return [r for r in results if r is not None]

    async def read_documents_async(
        self,
        file_paths: list[Path],
        file_extractor: dict | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[Document]:
        """Read documents from files concurrently.

        Args:
            file_paths: Paths to read
            file_extractor: Custom extractors by file extension
            on_progress: Optional progress callback

        Returns:
            List of loaded Documents
        """
        from llama_index.core import SimpleDirectoryReader

        def read_single_file(path: Path) -> list[Document]:
            try:
                reader = SimpleDirectoryReader(
                    input_files=[path],
                    file_extractor=file_extractor or {},
                )
                return list(reader.load_data(show_progress=False))
            except Exception as e:
                raise DocumentProcessingError(
                    f"Failed to read {path}: {e}",
                    file_path=str(path),
                    cause=e,
                )

        doc_lists = await self.map_async(
            read_single_file,
            file_paths,
            desc="Reading documents",
            on_progress=on_progress,
        )

        # Flatten the list of document lists
        return [doc for docs in doc_lists for doc in (docs or [])]

    async def embed_nodes_async(
        self,
        nodes: Sequence[BaseNode],
        embed_model: Any,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Sequence[BaseNode]:
        """Generate embeddings for nodes concurrently.

        Args:
            nodes: Nodes to embed
            embed_model: Embedding model
            on_progress: Optional progress callback

        Returns:
            Nodes with embeddings
        """
        total_batches = (len(nodes) + self.batch_size - 1) // self.batch_size
        completed_batches = 0

        async def embed_batch(batch: list[BaseNode]) -> list[BaseNode]:
            nonlocal completed_batches
            try:
                texts = [node.get_content() for node in batch]
                embeddings = embed_model.get_text_embedding_batch(texts)

                for node, embedding in zip(batch, embeddings):
                    node.embedding = embedding

                completed_batches += 1
                if on_progress:
                    on_progress(completed_batches, total_batches)

                return batch
            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
                completed_batches += 1
                if on_progress:
                    on_progress(completed_batches, total_batches)
                raise

        # Create batches
        batches = []
        for i in range(0, len(nodes), self.batch_size):
            batches.append(list(nodes[i : i + self.batch_size]))

        logger.info(f"Embedding {len(nodes)} nodes in {len(batches)} batches")

        # Process batches with limited concurrency
        semaphore = asyncio.Semaphore(self.max_workers or 4)

        async def limited_embed(batch: list[BaseNode]) -> list[BaseNode]:
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self._get_executor(), lambda: embed_batch_sync(batch, embed_model)
                )

        tasks = [limited_embed(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        embedded_nodes = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Batch embedding failed: {result}")
            else:
                embedded_nodes.extend(result)

        return embedded_nodes

    def close(self) -> None:
        """Close the executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "AsyncDocumentProcessor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> "AsyncDocumentProcessor":
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.close()


def embed_batch_sync(batch: list[BaseNode], embed_model: Any) -> list[BaseNode]:
    """Synchronous batch embedding for executor."""
    texts = [node.get_content() for node in batch]
    embeddings = embed_model.get_text_embedding_batch(texts)

    for node, embedding in zip(batch, embeddings):
        node.embedding = embedding

    return batch


async def process_documents_parallel(
    file_paths: list[Path],
    embed_model: Any,
    splitter: Any,
    max_workers: int | None = None,
    batch_size: int = 10,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> list[BaseNode]:
    """Process documents in parallel end-to-end.

    Convenience function that handles reading, splitting, and embedding
    documents concurrently.

    Args:
        file_paths: Paths to process
        embed_model: Embedding model
        splitter: Text splitter
        max_workers: Concurrent workers
        batch_size: Embedding batch size
        on_progress: Callback(stage, completed, total)

    Returns:
        List of processed and embedded nodes
    """
    async with AsyncDocumentProcessor(max_workers, batch_size=batch_size) as processor:
        # Read documents
        documents = await processor.read_documents_async(
            file_paths,
            on_progress=lambda c, t: (
                on_progress("reading", c, t) if on_progress else None
            ),
        )

        if not documents:
            return []

        # Split documents into nodes
        logger.info(f"Splitting {len(documents)} documents")
        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

        # Embed nodes
        embedded_nodes = await processor.embed_nodes_async(
            nodes,
            embed_model,
            on_progress=lambda c, t: (
                on_progress("embedding", c, t) if on_progress else None
            ),
        )

        return list(embedded_nodes)
