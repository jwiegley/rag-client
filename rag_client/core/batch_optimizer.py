"""Embedding batch size optimizer for RAG client.

This module provides automatic tuning of embedding batch sizes
for optimal throughput on different hardware configurations.
"""

import gc
import time
from dataclasses import dataclass
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BatchBenchmarkResult:
    """Result of a batch size benchmark."""

    batch_size: int
    throughput: float  # texts per second
    memory_mb: float  # peak memory usage
    latency_ms: float  # average latency per batch
    success: bool
    error: str | None = None


@dataclass
class OptimalBatchConfig:
    """Optimal batch configuration."""

    batch_size: int
    expected_throughput: float
    memory_headroom_mb: float

    def __str__(self) -> str:
        return (
            f"OptimalBatchConfig(batch_size={self.batch_size}, "
            f"throughput={self.expected_throughput:.1f} texts/s, "
            f"memory_headroom={self.memory_headroom_mb:.0f}MB)"
        )


class EmbeddingBatchOptimizer:
    """Optimizer for embedding batch sizes.

    Automatically determines the optimal batch size for embedding operations
    based on available memory and model characteristics. Uses binary search
    and benchmarking to find the best balance between throughput and stability.

    Example:
        >>> optimizer = EmbeddingBatchOptimizer(embed_model)
        >>> config = optimizer.find_optimal_batch_size()
        >>> print(f"Use batch_size={config.batch_size}")
    """

    DEFAULT_TEST_TEXTS = [
        "This is a sample text for benchmarking embedding performance.",
        "Machine learning models process text by converting it to vectors.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence has many applications in modern software.",
        "Document retrieval systems use semantic similarity for search.",
    ]

    def __init__(
        self,
        embed_model: Any,
        test_texts: list[str] | None = None,
        min_batch_size: int = 1,
        max_batch_size: int = 256,
        target_memory_usage: float = 0.7,  # Use 70% of available memory
    ):
        """Initialize batch optimizer.

        Args:
            embed_model: Embedding model to optimize for
            test_texts: Custom test texts (uses defaults if None)
            min_batch_size: Minimum batch size to test
            max_batch_size: Maximum batch size to test
            target_memory_usage: Target memory utilization (0-1)
        """
        self.embed_model = embed_model
        self.test_texts = test_texts or self.DEFAULT_TEST_TEXTS
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_usage = target_memory_usage

    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            return mem.available / (1024 * 1024)
        except ImportError:
            # Fallback to conservative estimate
            return 4096  # Assume 4GB available

    def get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0

    def benchmark_batch_size(
        self,
        batch_size: int,
        num_iterations: int = 3,
    ) -> BatchBenchmarkResult:
        """Benchmark a specific batch size.

        Args:
            batch_size: Batch size to test
            num_iterations: Number of iterations for averaging

        Returns:
            Benchmark results
        """
        texts = self.test_texts * ((batch_size // len(self.test_texts)) + 1)
        texts = texts[:batch_size]

        gc.collect()
        start_memory = self.get_current_memory_mb()

        latencies = []
        peak_memory = start_memory

        try:
            for _ in range(num_iterations):
                start_time = time.perf_counter()

                # Run embedding
                _ = self.embed_model.get_text_embedding_batch(texts)

                elapsed = (time.perf_counter() - start_time) * 1000  # ms
                latencies.append(elapsed)

                current_memory = self.get_current_memory_mb()
                peak_memory = max(peak_memory, current_memory)

            avg_latency = sum(latencies) / len(latencies)
            throughput = (batch_size / avg_latency) * 1000  # texts per second
            memory_used = peak_memory - start_memory

            return BatchBenchmarkResult(
                batch_size=batch_size,
                throughput=throughput,
                memory_mb=memory_used,
                latency_ms=avg_latency,
                success=True,
            )

        except Exception as e:
            return BatchBenchmarkResult(
                batch_size=batch_size,
                throughput=0,
                memory_mb=0,
                latency_ms=0,
                success=False,
                error=str(e),
            )
        finally:
            gc.collect()

    def find_max_stable_batch_size(self) -> int:
        """Find maximum batch size that doesn't cause OOM.

        Uses binary search to efficiently find the largest stable batch size.

        Returns:
            Maximum stable batch size
        """
        low = self.min_batch_size
        high = self.max_batch_size
        max_stable = low

        logger.info(f"Searching for max stable batch size in range [{low}, {high}]")

        while low <= high:
            mid = (low + high) // 2

            result = self.benchmark_batch_size(mid, num_iterations=1)

            if result.success:
                max_stable = mid
                low = mid + 1
                logger.debug(
                    f"  batch_size={mid}: OK (throughput={result.throughput:.1f})"
                )
            else:
                high = mid - 1
                logger.debug(f"  batch_size={mid}: FAILED ({result.error})")

        logger.info(f"Max stable batch size: {max_stable}")
        return max_stable

    def find_optimal_batch_size(
        self,
        test_sizes: list[int] | None = None,
    ) -> OptimalBatchConfig:
        """Find optimal batch size balancing throughput and memory.

        Tests multiple batch sizes and selects the one with best throughput
        while staying within memory constraints.

        Args:
            test_sizes: Specific sizes to test (auto-selects if None)

        Returns:
            Optimal batch configuration
        """
        # First find maximum stable size
        max_stable = self.find_max_stable_batch_size()

        # Generate test sizes if not provided
        if test_sizes is None:
            test_sizes = []
            size = self.min_batch_size
            while size <= max_stable:
                test_sizes.append(size)
                if size < 8:
                    size += 1
                elif size < 32:
                    size += 4
                elif size < 64:
                    size += 8
                else:
                    size += 16

        test_sizes = [s for s in test_sizes if s <= max_stable]

        logger.info(f"Benchmarking {len(test_sizes)} batch sizes")

        results = []
        for size in test_sizes:
            result = self.benchmark_batch_size(size)
            if result.success:
                results.append(result)
                logger.debug(
                    f"  batch_size={size}: {result.throughput:.1f} texts/s, "
                    f"{result.memory_mb:.0f}MB, {result.latency_ms:.1f}ms"
                )

        if not results:
            logger.warning("No successful benchmarks, using minimum batch size")
            return OptimalBatchConfig(
                batch_size=self.min_batch_size,
                expected_throughput=0,
                memory_headroom_mb=self.get_available_memory_mb(),
            )

        # Find best throughput within memory constraints
        available_memory = self.get_available_memory_mb() * self.target_memory_usage

        valid_results = [r for r in results if r.memory_mb < available_memory]
        if not valid_results:
            valid_results = results  # Fall back to all results

        # Select highest throughput
        best = max(valid_results, key=lambda r: r.throughput)

        logger.info(
            f"Optimal batch size: {best.batch_size} "
            f"({best.throughput:.1f} texts/s, {best.memory_mb:.0f}MB)"
        )

        return OptimalBatchConfig(
            batch_size=best.batch_size,
            expected_throughput=best.throughput,
            memory_headroom_mb=available_memory - best.memory_mb,
        )

    def get_recommended_batch_size(self, quick: bool = True) -> int:
        """Get recommended batch size with minimal testing.

        Args:
            quick: If True, uses heuristics instead of benchmarking

        Returns:
            Recommended batch size
        """
        if quick:
            # Use heuristics based on available memory
            available_mb = self.get_available_memory_mb()

            if available_mb > 16000:  # 16GB+
                return min(128, self.max_batch_size)
            elif available_mb > 8000:  # 8GB+
                return min(64, self.max_batch_size)
            elif available_mb > 4000:  # 4GB+
                return min(32, self.max_batch_size)
            elif available_mb > 2000:  # 2GB+
                return min(16, self.max_batch_size)
            else:
                return min(8, self.max_batch_size)

        # Full optimization
        config = self.find_optimal_batch_size()
        return config.batch_size


def auto_tune_batch_size(
    embed_model: Any,
    quick: bool = True,
) -> tuple[int, float]:
    """Auto-tune embedding batch size.

    Convenience function for quick batch size optimization.

    Args:
        embed_model: Embedding model
        quick: Use heuristics (True) or benchmark (False)

    Returns:
        Tuple of (batch_size, expected_throughput)
    """
    optimizer = EmbeddingBatchOptimizer(embed_model)

    if quick:
        batch_size = optimizer.get_recommended_batch_size(quick=True)
        return (batch_size, 0.0)

    config = optimizer.find_optimal_batch_size()
    return (config.batch_size, config.expected_throughput)
