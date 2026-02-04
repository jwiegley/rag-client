"""Retry utilities with exponential backoff for provider calls.

This module provides decorators and utilities for handling transient
failures when calling LLM and embedding providers.
"""

import asyncio
import functools
import logging
import random
import time
from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from ..exceptions import ProviderError, RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds to wait before next attempt
    """
    delay = min(base_delay * (2**attempt), max_delay)
    if jitter:
        delay = delay * (0.5 + random.random())
    return delay


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.

    Args:
        error: Exception to check

    Returns:
        True if the error should trigger a retry
    """
    if isinstance(error, RateLimitError):
        return True

    if isinstance(error, ProviderError):
        return True

    error_str = str(error).lower()
    retryable_patterns = [
        "connection",
        "timeout",
        "unavailable",
        "overloaded",
        "rate limit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "temporary",
        "transient",
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


def get_retry_after(error: Exception) -> Optional[float]:
    """Extract retry-after hint from error if available.

    Args:
        error: Exception to check

    Returns:
        Suggested retry delay in seconds, or None
    """
    if isinstance(error, RateLimitError) and error.context.get("retry_after"):
        return float(error.context["retry_after"])

    error_str = str(error)
    if "retry after" in error_str.lower():
        import re

        match = re.search(r"retry after (\d+)", error_str.lower())
        if match:
            return float(match.group(1))

    return None


class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay cap in seconds
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Exception types that trigger retries
        on_retry: Optional callback for retry events
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        on_retry: Optional[Callable[[Exception, int], None]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)
        self.on_retry = on_retry


DEFAULT_RETRY_CONFIG = RetryConfig()


def with_retry(
    config: Optional[RetryConfig] = None,
) -> Callable[[F], F]:
    """Decorator for adding retry logic to synchronous functions.

    Args:
        config: Retry configuration. Uses defaults if None.

    Returns:
        Decorated function with retry logic

    Example:
        >>> @with_retry(RetryConfig(max_retries=5))
        ... def call_api():
        ...     return requests.get("https://api.example.com")
    """
    cfg = config or DEFAULT_RETRY_CONFIG

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Optional[Exception] = None

            for attempt in range(cfg.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except cfg.retryable_exceptions as e:
                    last_error = e

                    if attempt >= cfg.max_retries:
                        logger.warning(
                            f"Max retries ({cfg.max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    if not is_retryable_error(e):
                        raise

                    retry_after = get_retry_after(e)
                    delay = retry_after or calculate_backoff(
                        attempt,
                        cfg.base_delay,
                        cfg.max_delay,
                        cfg.jitter,
                    )

                    logger.info(
                        f"Retry {attempt + 1}/{cfg.max_retries} for {func.__name__} "
                        f"after {delay:.2f}s due to: {e}"
                    )

                    if cfg.on_retry:
                        cfg.on_retry(e, attempt + 1)

                    time.sleep(delay)

            if last_error:
                raise last_error
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper  # type: ignore

    return decorator


def with_async_retry(
    config: Optional[RetryConfig] = None,
) -> Callable[[F], F]:
    """Decorator for adding retry logic to async functions.

    Args:
        config: Retry configuration. Uses defaults if None.

    Returns:
        Decorated async function with retry logic

    Example:
        >>> @with_async_retry(RetryConfig(max_retries=5))
        ... async def call_api():
        ...     async with aiohttp.ClientSession() as session:
        ...         return await session.get("https://api.example.com")
    """
    cfg = config or DEFAULT_RETRY_CONFIG

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Optional[Exception] = None

            for attempt in range(cfg.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except cfg.retryable_exceptions as e:
                    last_error = e

                    if attempt >= cfg.max_retries:
                        logger.warning(
                            f"Max retries ({cfg.max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    if not is_retryable_error(e):
                        raise

                    retry_after = get_retry_after(e)
                    delay = retry_after or calculate_backoff(
                        attempt,
                        cfg.base_delay,
                        cfg.max_delay,
                        cfg.jitter,
                    )

                    logger.info(
                        f"Retry {attempt + 1}/{cfg.max_retries} for {func.__name__} "
                        f"after {delay:.2f}s due to: {e}"
                    )

                    if cfg.on_retry:
                        cfg.on_retry(e, attempt + 1)

                    await asyncio.sleep(delay)

            if last_error:
                raise last_error
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper  # type: ignore

    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for provider health management.

    Prevents cascading failures by temporarily blocking calls to
    unhealthy providers.

    States:
        - CLOSED: Normal operation, calls pass through
        - OPEN: Provider unhealthy, calls fail fast
        - HALF_OPEN: Testing if provider has recovered

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=5)
        >>>
        >>> @breaker
        ... def call_provider():
        ...     return provider.complete("Hello")
        >>>
        >>> # After 5 failures, calls will fail fast
        >>> # After timeout, circuit becomes half-open for testing
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 1,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_requests: Successful requests needed to close
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None

    @property
    def state(self) -> str:
        """Get current circuit state."""
        if self._state == self.OPEN:
            if (
                self._last_failure_time
                and time.time() - self._last_failure_time >= self.recovery_timeout
            ):
                self._state = self.HALF_OPEN
                self._success_count = 0
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == self.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_requests:
                self._state = self.CLOSED
                self._failure_count = 0
        elif self._state == self.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures"
            )

    def __call__(self, func: F) -> F:
        """Decorator usage for circuit breaker."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.state == self.OPEN:
                raise ProviderError(
                    "Circuit breaker is open - provider temporarily unavailable"
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception:
                self.record_failure()
                raise

        return wrapper  # type: ignore


def retry_on_rate_limit(
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> Callable[[F], F]:
    """Convenience decorator specifically for rate limit handling.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff

    Returns:
        Decorated function
    """
    return with_retry(
        RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            retryable_exceptions=(RateLimitError, Exception),
        )
    )
