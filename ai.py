# !./.venv/bin/python

# ! /usr/bin/env nix-shell
# ! nix-shell -i python3 -p python3Packages.llama-index -p python3Packages.numpy_2 -p python3Packages.pypdf -p python3Packages.llama-index-llms-ollama -p python3Packages.llama-index-embeddings-huggingface -p python3Packages.llama-parse -p python3Packages.nltk -p python3Packages.orgparse

# python -m venv .venv
# source .venv/bin/activate
# pip install --upgrade pip
# pip install llama-index "numpy<2" pypdf llama-index-llms-ollama llama-index-embeddings-huggingface llama-parse nltk orgparse

import os
import sys
import asyncio
import logging
import yaml

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from copy import deepcopy
from functools import cache

from llama_index.cli.rag import RagCLI, default_ragcli_persist_dir
from llama_index.cli.rag.base import QueryPipelineQueryEngine, query_input
from llama_index.core import ChatPromptTemplate, Document
from llama_index.core import SimpleDirectoryReader, Settings, ChatPromptTemplate, PromptTemplate
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
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel

### Defaults

verbose = False

def enable_logging():
    global logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def setup_globals(llm = None):
    global Settings
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    if llm is not None:
        Settings.llm = llm

def get_ollama_model(
        embed_model,
        context_window: int = 32768,
        request_timeout: float = 3600.0,
        temperature: float = 0.9,
        base_url: str = "http://127.0.0.1:11434/"):
    return Ollama(
        model=embed_model,
        context_window=context_window,
        request_timeout=request_timeout,
        temperature=temperature,
        base_url=base_url
    )

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

def build_index_from_directory(directory, llm = None):
    documents = SimpleDirectoryReader(
        directory,
        file_extractor=file_extractor,
        recursive=True
    ).load_data()
    return VectorStoreIndex.from_documents(
        documents,
        llm=llm,
        embed_model=Settings.embed_model,
        show_progress=verbose,
    )

def load_rag_index_of_directory(directory, persist_dir, llm = None):
    if os.path.exists(persist_dir):
        return read_index_from_cache(persist_dir)
    else:
        index = build_index_from_directory(directory)
        write_index_to_cache(index, persist_dir)
        return index

### Retrieving

### Prompts

def prompt_templates(basic_prompt):
    qa_prompt_str = '''
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    ''' + basic_prompt + '''
    Given the context information and not prior knowledge,
    answer the question: {query_str}
    '''

    refine_prompt_str = '''
    We have the opportunity to refine the original answer (only if needed) with
    some more context below.
    ------------
    {context_msg}
    ------------
    ''' + basic_prompt + '''
    Given the new context, refine the original answer to better answer the
    question: {query_str}. If the context isn't useful, output the original answer
    again.
    Original Answer: {existing_answer}"
    '''

    chat_text_qa_msgs = [
        (
            "system",
            "Always answer the question, even if the context isn't helpful.",
        ),
        ("user", qa_prompt_str),
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    chat_refine_msgs = [
        (
            "system",
            "Always answer the question, even if the context isn't helpful.",
        ),
        ("user", refine_prompt_str),
    ]
    refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

    return (text_qa_template, refine_template)

### Querying

def submit_query(
        index,
        llm,
        query_text: str,
        prompt_text: str = "You are a helpful automated assistant who answers clearly and succinctly.",
        similarity_top_k: int = 8
):
    (text_qa_template, refine_template) = prompt_templates(prompt_text)
    query_engine = index.as_query_engine(
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        llm=llm,
        streaming=True,
        similarity_top_k=similarity_top_k,
        verbose=verbose
    )
    return query_engine.query(query_text)

def submit_query_with_file(
        index,
        llm,
        query_text: str,
        file_path: str,
        prompt_text: str = "You are a helpful automated assistant who answers clearly and succinctly.",
        similarity_top_k: int = 8
):
    with open(file_path, 'r') as file:
        return submit_query(
            index,
            llm=llm,
            query_text=query_text + file.read(),
            prompt_text=prompt_text,
            similarity_top_k=similarity_top_k
        )

def query_models_with_file(
        index,
        query_models: List[str],
        query_text: str,
        file_path: str,
        prompt_text: str = "You are a helpful automated assistant who answers clearly and succinctly.",
        similarity_top_k: int = 8
):
    for model in query_models:
        print("")
        print("")
        print("# Model: ", model)
        print("")
        submit_query_with_file(
            index,
            llm=get_ollama_model(model),
            query_text=query_text,
            file_path=file_path,
            prompt_text=prompt_text,
            similarity_top_k=similarity_top_k
        ).print_response_stream()

##############################################################################

with open(sys.argv[1]) as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

print("Building RAG index...")

llm = get_ollama_model(
    embed_model=config['embed_model'],
    context_window=config['context_window'],
    request_timeout=config['request_timeout'],
    temperature=config['temperature'],
    base_url=config['base_url'],
)

setup_globals(llm)

index = load_rag_index_of_directory(
    directory=config['directory'],
    persist_dir=config['persist_dir'],
    llm=llm,
)

print("Submitting query...")

query_models_with_file(
    index=index,
    query_models=config['query_models'],
    query_text=config['query_text'],
    file_path=sys.argv[2],
    prompt_text=config['prompt_text'],
    similarity_top_k=config['similarity_top_k'],
)
