"""
Parallel processing utilities for preprocessing pipeline.

This module provides thread-safe parallel execution with rate limiting
for processing Confluence attachments and metadata extraction.

Features:
- ThreadPoolExecutor wrapper with configurable workers
- Token bucket rate limiting for API calls
- Error collection without failing entire batch
- Progress tracking with optional logging

Example:
    >>> from preprocessing.parallel import ParallelProcessor, RateLimiter
    >>> processor = ParallelProcessor(max_workers=4, rate_limit_rps=10.0)
    >>> results = processor.map(process_func, items)
    >>> for result in results:
    ...     if result.success:
    ...         print(result.value)
    ...     else:
    ...         print(f"Error: {result.error}")
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterator, List, Optional, TypeVar

from loguru import logger

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ProcessingResult(Generic[T]):
    """Result wrapper for parallel processing.

    Attributes:
        item: Original input item
        value: Result value if successful
        success: Whether processing succeeded
        error: Exception if failed
        duration_ms: Processing time in milliseconds
    """

    item: Any
    value: Optional[T] = None
    success: bool = False
    error: Optional[Exception] = None
    duration_ms: float = 0.0


class RateLimiter:
    """Token bucket rate limiter for controlling API call frequency.

    Implements a token bucket algorithm where tokens are added at a fixed
    rate and consumed by each request. If no tokens are available, the
    caller blocks until one becomes available.

    Attributes:
        requests_per_second: Maximum requests per second
        tokens: Current number of available tokens
        max_tokens: Maximum token capacity (burst limit)

    Example:
        >>> limiter = RateLimiter(requests_per_second=10.0)
        >>> for item in items:
        ...     limiter.acquire()  # Blocks if rate exceeded
        ...     process(item)
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_multiplier: float = 1.5,
    ) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_second: Target RPS limit
            burst_multiplier: Allow burst up to this multiple of RPS
        """
        self.rps = requests_per_second
        self.max_tokens = requests_per_second * burst_multiplier
        self.tokens = self.max_tokens
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rps)

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire a token, blocking if necessary.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if token acquired, False if timeout exceeded
        """
        start_time = time.monotonic()

        while True:
            with self._lock:
                self._refill_tokens()

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True

                # Calculate wait time for next token
                wait_time = (1.0 - self.tokens) / self.rps

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    return False

            # Sleep until we expect to have a token
            time.sleep(min(wait_time, 0.1))  # Cap sleep to allow checking

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token acquired, False otherwise
        """
        with self._lock:
            self._refill_tokens()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


class ParallelProcessor:
    """Parallel task processor with rate limiting and error handling.

    Executes tasks concurrently using a thread pool while respecting
    rate limits and collecting errors without failing the entire batch.

    Attributes:
        max_workers: Maximum concurrent threads
        rate_limiter: Optional rate limiter for API calls
        executor: ThreadPoolExecutor instance

    Example:
        >>> processor = ParallelProcessor(max_workers=4, rate_limit_rps=10.0)
        >>> results = processor.map(analyze_image, image_paths)
        >>> successful = [r.value for r in results if r.success]
        >>> failed = [r for r in results if not r.success]
    """

    def __init__(
        self,
        max_workers: int = 4,
        rate_limit_rps: Optional[float] = None,
    ) -> None:
        """Initialize parallel processor.

        Args:
            max_workers: Maximum concurrent threads
            rate_limit_rps: Optional requests per second limit
        """
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(rate_limit_rps) if rate_limit_rps else None
        self._executor: Optional[ThreadPoolExecutor] = None
        logger.debug(
            f"ParallelProcessor initialized: workers={max_workers}, "
            f"rps_limit={rate_limit_rps}"
        )

    def __enter__(self) -> "ParallelProcessor":
        """Context manager entry."""
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - shutdown executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def _process_item(
        self,
        func: Callable[[T], R],
        item: T,
    ) -> ProcessingResult[R]:
        """Process a single item with rate limiting.

        Args:
            func: Processing function
            item: Item to process

        Returns:
            ProcessingResult with value or error
        """
        # Apply rate limiting
        if self.rate_limiter:
            self.rate_limiter.acquire()

        start_time = time.monotonic()

        try:
            value = func(item)
            duration_ms = (time.monotonic() - start_time) * 1000
            return ProcessingResult(
                item=item,
                value=value,
                success=True,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.warning(f"Processing failed for item: {e}")
            return ProcessingResult(
                item=item,
                success=False,
                error=e,
                duration_ms=duration_ms,
            )

    def map(
        self,
        func: Callable[[T], R],
        items: List[T],
        desc: Optional[str] = None,
    ) -> List[ProcessingResult[R]]:
        """Process items in parallel and collect results.

        Args:
            func: Function to apply to each item
            items: List of items to process
            desc: Optional description for logging

        Returns:
            List of ProcessingResult objects in original order
        """
        if not items:
            return []

        desc = desc or "Processing"
        logger.info(f"{desc}: {len(items)} items with {self.max_workers} workers")

        executor = self._get_executor()
        results: List[ProcessingResult[R]] = [None] * len(items)  # type: ignore

        # Submit all tasks
        future_to_index = {
            executor.submit(self._process_item, func, item): i
            for i, item in enumerate(items)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
            completed += 1

            # Log progress every 10%
            if completed % max(1, len(items) // 10) == 0:
                logger.debug(f"{desc}: {completed}/{len(items)} completed")

        # Summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_time = sum(r.duration_ms for r in results)
        avg_time = total_time / len(results) if results else 0

        logger.info(
            f"{desc} complete: {successful} successful, {failed} failed, "
            f"avg {avg_time:.0f}ms per item"
        )

        return results

    def map_batched(
        self,
        func: Callable[[T], R],
        items: List[T],
        batch_size: int = 50,
        desc: Optional[str] = None,
        pause_between_batches: float = 1.0,
    ) -> List[ProcessingResult[R]]:
        """Process items in batches with pause between batches.

        Useful for large workloads where you want to give the API
        time to recover between batches.

        Args:
            func: Function to apply to each item
            items: List of items to process
            batch_size: Number of items per batch
            desc: Optional description for logging
            pause_between_batches: Seconds to pause between batches

        Returns:
            List of ProcessingResult objects in original order
        """
        if not items:
            return []

        desc = desc or "Batch processing"
        all_results: List[ProcessingResult[R]] = []
        num_batches = (len(items) + batch_size - 1) // batch_size

        logger.info(
            f"{desc}: {len(items)} items in {num_batches} batches of {batch_size}"
        )

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(items))
            batch = items[start_idx:end_idx]

            batch_desc = f"{desc} [batch {batch_idx + 1}/{num_batches}]"
            results = self.map(func, batch, desc=batch_desc)
            all_results.extend(results)

            # Pause between batches (but not after the last one)
            if batch_idx < num_batches - 1 and pause_between_batches > 0:
                logger.debug(f"Pausing {pause_between_batches}s before next batch")
                time.sleep(pause_between_batches)

        return all_results

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None


def batched(items: List[T], batch_size: int) -> Iterator[List[T]]:
    """Split items into batches.

    Args:
        items: List to split
        batch_size: Maximum items per batch

    Yields:
        Lists of at most batch_size items
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
