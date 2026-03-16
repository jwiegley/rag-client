"""Incremental document caching for RAG client.

This module provides per-document fingerprinting and caching to enable
incremental indexing - only re-indexing documents that have changed.
"""

import hashlib
import json
import pickle
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.schema import BaseNode

from ..utils.helpers import cache_dir


@dataclass
class DocumentCacheEntry:
    """Cache entry for a single document.

    Attributes:
        file_path: Absolute path to the source document
        content_hash: SHA-256 hash of file contents
        mtime: Last modification time of file
        size: File size in bytes
        node_ids: List of node IDs generated from this document
        cached_at: Timestamp when cache entry was created
        config_hash: Hash of relevant config settings
    """

    file_path: str
    content_hash: str
    mtime: float
    size: int
    node_ids: list[str]
    cached_at: float
    config_hash: str


@dataclass
class CacheManifest:
    """Manifest tracking all cached documents.

    Attributes:
        version: Cache format version for compatibility
        entries: Map of file paths to cache entries
        config_hash: Hash of configuration affecting indexing
        created_at: Timestamp when manifest was created
        updated_at: Timestamp of last update
    """

    version: int = 1
    entries: dict[str, DocumentCacheEntry] = field(default_factory=dict)
    config_hash: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class DocumentCache:
    """Manages incremental document caching for efficient re-indexing.

    This class enables incremental indexing by tracking individual document
    fingerprints. When re-indexing, only documents that have changed need
    to be processed, significantly reducing indexing time for large collections.

    Features:
        - Per-document content hashing (SHA-256)
        - Configuration-aware caching (re-index on config change)
        - Automatic cache invalidation for modified files
        - Node storage for quick retrieval of unchanged documents

    Example:
        >>> cache = DocumentCache(cache_name="my_collection")
        >>>
        >>> # Check which files need indexing
        >>> changed, unchanged = cache.check_files(file_paths, config_hash)
        >>>
        >>> # Index only changed files, get cached nodes for unchanged
        >>> for path in unchanged:
        ...     nodes.extend(cache.get_nodes(path))
        >>>
        >>> # Update cache with newly indexed files
        >>> cache.update_entry(path, content_hash, node_ids, config_hash)
        >>> cache.save_manifest()
    """

    def __init__(
        self,
        cache_name: str = "default",
        cache_base: Path | None = None,
    ):
        """Initialize document cache.

        Args:
            cache_name: Name for this cache collection
            cache_base: Base directory for cache storage.
                       Defaults to ~/.cache/rag-client/
        """
        self.cache_base = cache_base or cache_dir()
        self.cache_name = cache_name
        self.cache_path = self.cache_base / cache_name
        self.cache_path.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.cache_path / "manifest.json"
        self.nodes_path = self.cache_path / "nodes"
        self.nodes_path.mkdir(exist_ok=True)

        self.manifest = self._load_manifest()

    def _load_manifest(self) -> CacheManifest:
        """Load cache manifest from disk."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    data = json.load(f)

                entries = {}
                for path, entry_data in data.get("entries", {}).items():
                    entries[path] = DocumentCacheEntry(**entry_data)

                return CacheManifest(
                    version=data.get("version", 1),
                    entries=entries,
                    config_hash=data.get("config_hash", ""),
                    created_at=data.get("created_at", time.time()),
                    updated_at=data.get("updated_at", time.time()),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                return CacheManifest()
        return CacheManifest()

    def save_manifest(self) -> None:
        """Save cache manifest to disk."""
        self.manifest.updated_at = time.time()
        data = {
            "version": self.manifest.version,
            "entries": {
                path: asdict(entry) for path, entry in self.manifest.entries.items()
            },
            "config_hash": self.manifest.config_hash,
            "created_at": self.manifest.created_at,
            "updated_at": self.manifest.updated_at,
        }
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of file contents.

        Args:
            file_path: Path to file

        Returns:
            Hex-encoded SHA-256 hash
        """
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_config_hash(config: Any) -> str:
        """Compute hash of configuration affecting indexing.

        Args:
            config: Configuration object (will be repr'd)

        Returns:
            Hex-encoded SHA-256 hash of config repr
        """
        config_str = repr(config)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def check_files(
        self,
        file_paths: list[Path],
        config_hash: str,
    ) -> tuple[list[Path], list[Path]]:
        """Check which files need to be re-indexed.

        Compares file fingerprints against cache to determine which
        files have changed and need re-indexing.

        Args:
            file_paths: List of file paths to check
            config_hash: Hash of current indexing configuration

        Returns:
            Tuple of (changed_files, unchanged_files)
            - changed_files: Need to be re-indexed
            - unchanged_files: Can use cached nodes
        """
        if self.manifest.config_hash != config_hash:
            return (file_paths, [])

        changed = []
        unchanged = []

        for file_path in file_paths:
            path_str = str(file_path.absolute())

            if path_str not in self.manifest.entries:
                changed.append(file_path)
                continue

            entry = self.manifest.entries[path_str]

            try:
                stat = file_path.stat()
                if stat.st_mtime != entry.mtime or stat.st_size != entry.size:
                    current_hash = self.compute_file_hash(file_path)
                    if current_hash != entry.content_hash:
                        changed.append(file_path)
                    else:
                        entry.mtime = stat.st_mtime
                        unchanged.append(file_path)
                else:
                    unchanged.append(file_path)
            except OSError:
                changed.append(file_path)

        return (changed, unchanged)

    def get_nodes(self, file_path: Path) -> list[BaseNode]:
        """Retrieve cached nodes for a document.

        Args:
            file_path: Path to source document

        Returns:
            List of cached nodes, or empty list if not cached
        """
        path_str = str(file_path.absolute())
        if path_str not in self.manifest.entries:
            return []

        entry = self.manifest.entries[path_str]
        nodes = []

        for node_id in entry.node_ids:
            node_path = self.nodes_path / f"{node_id}.pkl"
            if node_path.exists():
                try:
                    with open(node_path, "rb") as f:
                        nodes.append(pickle.load(f))
                except (pickle.PickleError, OSError):
                    return []

        return nodes

    def cache_nodes(
        self,
        file_path: Path,
        nodes: Sequence[BaseNode],
        config_hash: str,
    ) -> None:
        """Cache nodes for a document.

        Args:
            file_path: Path to source document
            nodes: Nodes generated from the document
            config_hash: Hash of indexing configuration
        """
        path_str = str(file_path.absolute())
        stat = file_path.stat()
        content_hash = self.compute_file_hash(file_path)

        node_ids = []
        for node in nodes:
            node_id = node.node_id
            node_ids.append(node_id)
            node_path = self.nodes_path / f"{node_id}.pkl"
            with open(node_path, "wb") as f:
                pickle.dump(node, f)

        self.manifest.entries[path_str] = DocumentCacheEntry(
            file_path=path_str,
            content_hash=content_hash,
            mtime=stat.st_mtime,
            size=stat.st_size,
            node_ids=node_ids,
            cached_at=time.time(),
            config_hash=config_hash,
        )
        self.manifest.config_hash = config_hash

    def invalidate(self, file_paths: list[Path] | None = None) -> None:
        """Invalidate cache entries.

        Args:
            file_paths: Specific files to invalidate. If None, clears all.
        """
        if file_paths is None:
            for entry in self.manifest.entries.values():
                for node_id in entry.node_ids:
                    node_path = self.nodes_path / f"{node_id}.pkl"
                    if node_path.exists():
                        node_path.unlink()
            self.manifest.entries.clear()
        else:
            for file_path in file_paths:
                path_str = str(file_path.absolute())
                if path_str in self.manifest.entries:
                    entry = self.manifest.entries[path_str]
                    for node_id in entry.node_ids:
                        node_path = self.nodes_path / f"{node_id}.pkl"
                        if node_path.exists():
                            node_path.unlink()
                    del self.manifest.entries[path_str]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_nodes = sum(len(e.node_ids) for e in self.manifest.entries.values())
        return {
            "cached_documents": len(self.manifest.entries),
            "total_nodes": total_nodes,
            "cache_size_mb": sum(
                (self.nodes_path / f"{nid}.pkl").stat().st_size
                for e in self.manifest.entries.values()
                for nid in e.node_ids
                if (self.nodes_path / f"{nid}.pkl").exists()
            )
            / (1024 * 1024),
            "config_hash": self.manifest.config_hash,
            "last_updated": self.manifest.updated_at,
        }
