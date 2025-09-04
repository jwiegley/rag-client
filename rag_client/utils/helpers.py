"""Utility functions and helper classes for RAG client."""

import hashlib
import os
import sys
from pathlib import Path
from typing import NoReturn

from xdg_base_dirs import xdg_cache_home


def error(msg: str) -> NoReturn:
    """Print error message and exit.
    
    Args:
        msg: Error message to display
    """
    print(msg, file=sys.stderr)
    sys.exit(1)


def parse_prefixes(prefixes: list[str], s: str) -> tuple[str | None, str]:
    """Parse a string for matching prefixes.
    
    Args:
        prefixes: List of prefixes to check
        s: String to parse
        
    Returns:
        Tuple of (matched_prefix, remainder) or (None, original_string)
    """
    for prefix in prefixes:
        if s.startswith(prefix):
            return prefix, s[len(prefix):]
    return None, s  # No matching prefix found


def list_files(directory: Path, recursive: bool = False) -> list[Path]:
    """List files in a directory.
    
    Args:
        directory: Directory path
        recursive: Whether to recurse into subdirectories
        
    Returns:
        List of file paths
    """
    if recursive:
        file_list: list[Path] = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file not in [".", ".."]:
                    file_list.append(Path(root) / Path(file))
        return file_list
    else:
        return [
            directory / f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]


def read_files(
    read_from: str,
    recursive: bool = False,
) -> list[Path] | NoReturn:
    """Read file paths from various sources.
    
    Args:
        read_from: Source specification (-, directory path, or file path)
        recursive: Whether to recurse into subdirectories
        
    Returns:
        List of file paths
        
    Raises:
        SystemExit: If input is invalid
    """
    if read_from == "-":
        input_files: list[str] = [line.strip() for line in sys.stdin if line.strip()]
        if not input_files:
            error("No filenames provided on standard input")
        all_files = [read_files(path, recursive) for path in input_files]
        return [item for sublist in all_files for item in sublist]
    elif os.path.isdir(read_from):
        return list_files(Path(read_from), recursive)
    elif os.path.isfile(read_from):
        return [Path(read_from)]
    else:
        error(f"Input path is unrecognized or non-existent: {read_from}")


def convert_str(read_from: str | None) -> str | None:
    """Convert input source to string content.
    
    Args:
        read_from: Input source (None, -, file path, or direct string)
        
    Returns:
        String content or None
        
    Raises:
        SystemExit: If no input provided on stdin when expected
    """
    if read_from is None:
        return read_from
    elif read_from == "-":
        s = sys.stdin.read()
        if not s:
            error("No input provided on standard input")
        return s
    elif os.path.isfile(read_from):
        with open(read_from, "r") as f:
            return f.read()
    else:
        return read_from


def collection_hash(file_list: list[Path]) -> str:
    """Compute a hash of a collection of files.
    
    Args:
        file_list: List of file paths
        
    Returns:
        SHA-512 hash of the concatenated file hashes
    """
    # List to hold the hash of each file
    file_hashes: list[str] = []
    for file_path in file_list:
        # Compute SHA-512 hash of the file contents
        h = hashlib.sha512()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        file_hashes.append(h.hexdigest())
    # Concatenate all hashes with newline separators
    concatenated = "\n".join(file_hashes).encode("utf-8")
    # Compute SHA-512 hash of the concatenated hashes
    final_hash = hashlib.sha512(concatenated).hexdigest()
    return final_hash


def cache_dir() -> Path:
    """Get the cache directory for rag-client.
    
    Returns:
        Path to cache directory (created if doesn't exist)
    """
    d = xdg_cache_home() / "rag-client"
    d.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    return d


def clean_special_tokens(text: str) -> str:
    """Remove special tokens from text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove <|assistant|> with various newline combinations
    patterns = [
        "<|assistant|>\n\n",
        "<|assistant|>\n",
        "\n\n<|assistant|>",
        "\n<|assistant|>",
        "<|assistant|>",
    ]
    for pattern in patterns:
        text = text.replace(pattern, "")
    return text