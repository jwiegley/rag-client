# !./.venv/bin/python

# python -m venv .venv
# source .venv/bin/activate
# pip install --upgrade pip

# pip install llama-index "numpy<2" pypdf llama-parse nltk orgparse \
#     llama-index-embeddings-huggingface 

import os
import sys
import asyncio
import logging
import yaml
import argparse
import json

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from copy import deepcopy
from functools import cache

from llama_index.core import ChatPromptTemplate, Document
from llama_index.core import SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.extractors import TitleExtractor
from llama_index.core.indices import load_index_from_storage
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import CustomQueryEngine, RetrieverQueryEngine
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.query_pipeline.components.function import FnComponent
from llama_index.core.readers.base import BaseReader
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.retrievers import VectorIndexRetriever, VectorContextRetriever
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel

### Defaults

verbose = False

def enable_logging():
    global logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def setup_globals(embed_model, chunk_size = 512, chunk_overlap = 50):
    global Settings
    Settings.embed_model = embed_model
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap

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


class OrgReader(BaseReader, BaseModel):
    """OrgReader

    Extract text from org files.
    Add the :PROPERTIES: on text node as extra_info
    """

    split_depth: int = 0
    text_formatting: str = "plain"  # plain or raw, as supported by orgparse

    def node_to_document(self, node, extra_info: Optional[Dict] = None) -> Document:
        """Convert org node to document."""
        text = "\n".join(get_text_from_org_node(node, format=self.text_formatting))
        extra_info = deepcopy(extra_info or {})
        for prop, value in node.properties.items():
            extra_info["org_property_" + prop] = value
        return Document(text=text, extra_info=extra_info)

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file into different documents based on root depth."""
        from orgparse import OrgNode, load

        org_content: OrgNode = load(file)
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

### Tools

### Ingestion

def read_index_from_cache(persist_dir):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index

def write_index_to_cache(index, persist_dir):
    index.storage_context.persist(
        persist_dir=persist_dir
    )

def build_index_from_directory(directory):
    documents = SimpleDirectoryReader(
        directory,
        file_extractor=file_extractor,
        recursive=True
    ).load_data()
    return VectorStoreIndex.from_documents(
        documents,
        embed_model=Settings.embed_model,
        show_progress=verbose,
    )

def load_rag_index_of_directory(directory, persist_dir):
    if os.path.exists(persist_dir):
        return read_index_from_cache(persist_dir)
    else:
        index = build_index_from_directory(directory)
        write_index_to_cache(index, persist_dir)
        return index

def main():
    parser = argparse.ArgumentParser(description='Read long command-line options')
    parser.add_argument('--embed-model', type=str, help='Embedding model')
    parser.add_argument('--top-k', type=int, help='Top K document nodes')
    parser.add_argument('--directory', type=str, help='Directory')
    parser.add_argument('--persist-dir', type=str, help='Persist dir')
    parser.add_argument('--verbose', action="store_true", help='Verbose?')
    parser.add_argument('--query', type=str, help='Query text')
    args = parser.parse_args()

    global verbose
    verbose = args.verbose

    embed_model = HuggingFaceEmbedding(model_name=args.embed_model)
    
    setup_globals(embed_model=embed_model)
    
    index = load_rag_index_of_directory(
        directory=args.directory,
        persist_dir=args.persist_dir,
    )

    retriever = index.as_retriever(similarity_top_k=args.top_k)

    nodes = retriever.retrieve(args.query)

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
