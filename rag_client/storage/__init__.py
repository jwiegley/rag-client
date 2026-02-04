"""Storage backends for RAG client.

This module provides storage implementations for document caching,
vector storage, and persistence.
"""

from .cache import CacheManifest, DocumentCache, DocumentCacheEntry

__all__ = [
    "DocumentCache",
    "DocumentCacheEntry",
    "CacheManifest",
]
