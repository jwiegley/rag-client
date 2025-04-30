#!/usr/bin/env python

import argparse
import json
import os
import sys
import hashlib
import base64

from copy import deepcopy
from functools import cache
from typing import List
from xdg_base_dirs import xdg_cache_home

from llama_index.core import Document, Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex
from llama_index.core.indices import load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def collection_hash(file_list):
    # List to hold the hash of each file
    file_hashes = []
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

### Readers

@cache
def get_text_from_org_node(current_node, format: str = "plain") -> List[str]:
    """Extract text from org node. Skip properties"""
    lines = []
    if current_node.heading:
        lines.append(current_node.get_heading(format=format))
    if current_node.body:
        lines.extend(current_node.get_body(format=format).split("\n"))
    for child in current_node.children:
        lines.extend(get_text_from_org_node(child, format=format))

    return lines


class OrgReader(BaseReader):
    """OrgReader

    Extract text from org files.
    Add the :PROPERTIES: on text node as extra_info
    """

    split_depth: int = 0
    text_formatting: str = "plain"  # plain or raw, as supported by orgparse

    def node_to_document(self, node, extra_info):
        """Convert org node to document."""
        text = "\n".join(get_text_from_org_node(node, format=self.text_formatting))
        extra_info = deepcopy(extra_info or {})
        for prop, value in node.properties.items():
            extra_info["org_property_" + prop] = value
        return Document(text=text, extra_info=extra_info)

    def load_data(self, file, extra_info):
        """Parse file into different documents based on root depth."""
        from orgparse import load

        org_content = load(file)
        documents: List[Document] = []

        extra_info = extra_info or {}
        extra_info["filename"] = org_content.env.filename

        # In orgparse, list(org_content) ALL the nodes in the file
        # So we use this to process the nodes below the split_depth as
        # separate documents and skip the rest. This means at a split_depth
        # of 2, we make documents from nodes at levels 0 (whole file), 1, and 2.
        # The text will be present in multiple documents!
        for node in list(org_content):
            if node.level <= self.split_depth:
                documents.append(self.node_to_document(node, extra_info))

        return documents

file_extractor = {".org": OrgReader()}

### Ingestion

def load_files_index(input_files,
                     embed_model,
                     chunk_size = None,
                     chunk_overlap = None,
                     verbose = False):
    # Determine the cache directory as a unique function of the inputs
    cache_dir = xdg_cache_home() / "rag-client"
    cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    fingerprint = [
        collection_hash(input_files),
        hashlib.sha512(embed_model.encode("utf-8")).hexdigest(),
        hashlib.sha512(str(chunk_size).encode("utf-8")).hexdigest(),
        hashlib.sha512(str(chunk_overlap).encode("utf-8")).hexdigest(),
    ]
    final_hash = "\n".join(fingerprint).encode("utf-8")
    final_base64 = base64.b64encode(final_hash).decode('utf-8')[0:32]

    persist_dir = cache_dir / final_base64
    if verbose: print(f'Cache directory = {persist_dir}')

    embed_model = HuggingFaceEmbedding(model_name=embed_model)

    if persist_dir is not None and os.path.exists(persist_dir):
        # If a cache dir was specified and exists, load the index
        global Settings
        Settings.embed_model = embed_model
        if chunk_size is not None:
            Settings.chunk_size = chunk_size
        if chunk_overlap is not None:
            Settings.chunk_overlap = chunk_overlap
        if verbose: print('Load index from cache')
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(persist_dir))
        )
    else:
        if verbose: print('Read documents from disk')
        documents = SimpleDirectoryReader(
            input_files=input_files,
            file_extractor=file_extractor,  # type: ignore[override]
            recursive=True
        ).load_data()

        if verbose: print('Chunk documents into semantic units')
        if chunk_size is not None and chunk_overlap is not None:
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            nodes = splitter.get_nodes_from_documents(documents)
        else:
            nodes = SimpleNodeParser().get_nodes_from_documents(documents)

        if verbose: print('Calculate vector embeddings')
        index = VectorStoreIndex(
            nodes,
            embed_model=embed_model,
            show_progress=verbose,
        )

        # If a cache dir was specified, persist the index
        if persist_dir is not None:
            if verbose: print('Write index to cache')
            index.storage_context.persist(
                persist_dir=persist_dir
            )

    return index

def main():
    parser = argparse.ArgumentParser(description='Read long command-line options')
    parser.add_argument('--embed-model', type=str, help='Embedding model')
    parser.add_argument('--embed-dim', type=int, help='Embedding dimensions')
    parser.add_argument('--chunk-size', type=int, help='Chunk size')
    parser.add_argument('--chunk-overlap', type=int, help='Chunk overlap')
    parser.add_argument('--top-k', type=int, help='Top K document nodes')
    parser.add_argument('--directory', type=str, help='Directory')
    parser.add_argument('--verbose', action="store_true", help='Verbose?')
    parser.add_argument('--query', type=str, help='Query text')
    args = parser.parse_args()
    
    input_files = [line.strip() for line in sys.stdin if line.strip()]
    if not input_files:
        print("No filenames provided on standard input.", file=sys.stderr)
        sys.exit(1)

    index = load_files_index(
        input_files=input_files,
        embed_model=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        verbose=args.verbose,
        )

    if args.verbose: print('Create retriever object')
    retriever = index.as_retriever(similarity_top_k=args.top_k)

    if args.verbose: print('Query retriever')
    nodes = retriever.retrieve(args.query)

    if args.verbose: print('Format output')
    node_dicts = []
    for node in nodes:
        node_dict = {
            "text": node.text,
            "metadata": node.metadata,
            "score": node.score,
            "id": node.id_
        }
        node_dicts.append(node_dict)

    json_output = json.dumps(node_dicts, indent=2)
    print(json_output)

if __name__ == '__main__':
    main()
