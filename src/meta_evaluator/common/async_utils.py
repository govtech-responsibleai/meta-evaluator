"""Async utilities for MetaEvaluator."""

import asyncio
import functools
from typing import Any, Callable


def sync_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to create synchronous wrapper for async functions.

    This decorator allows users to call async methods synchronously without
    having to use await or manage asyncio themselves. The async execution
    is handled internally using asyncio.run().

    Args:
        func: The async function to wrap.

    Returns:
        A synchronous function that internally calls asyncio.run() on the async function.

    Example:
        >>> @sync_wrapper
        >>> def my_sync_method(self, *args, **kwargs):
        >>>     return self._my_async_method(*args, **kwargs)
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return asyncio.run(func(self, *args, **kwargs))

    return wrapper
