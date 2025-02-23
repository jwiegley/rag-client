# !./.venv/bin/python

# ! /usr/bin/env nix-shell
# ! nix-shell -i python3 -p python3Packages.llama-index -p python3Packages.numpy_2 -p python3Packages.chromadb -p python3Packages.llama-index-vector-stores-chroma -p python3Packages.pypdf -p python3Packages.llama-index-llms-ollama -p python3Packages.llama-index-embeddings-huggingface -p python3Packages.llama-parse -p python3Packages.nltk -p python3Packages.orgparse

# python -m venv .venv
# source .venv/bin/activate
# pip install --upgrade pip
# pip install llama-index "numpy<2" chromadb llama-index-vector-stores-chroma pypdf llama-index-llms-ollama llama-index-embeddings-huggingface nltk orgparse

import os
import asyncio
import chromadb
# import qdrant_client

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from copy import deepcopy
from functools import cache

from llama_index.cli.rag import RagCLI, default_ragcli_persist_dir
from llama_index.cli.rag.base import QueryPipelineQueryEngine, query_input
from llama_index.core import ChatPromptTemplate
from llama_index.core import Document
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
from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel

verbose = True

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

### Defaults

def setup_model(model, context_window: int = 32768):
    llm = Ollama(
        model=model,
        context_window=32768,
        request_timeout=300.0,
        temperature=0.9,
        base_url="http://127.0.0.1:11434/"
    )

    global Settings
    Settings.llm = llm

    return llm

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

PERSIST_DIR = "./storage"

## CHROMA
# chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
# chroma_collection = chroma_client.create_collection("default", get_or_create=True)
# vector_store = ChromaVectorStore(
#     chroma_collection=chroma_collection,
#     persist_dir=PERSIST_DIR
# )

## QDRANT
# client = qdrant_client.QdrantClient(location=":memory:")
# vector_store = QdrantVectorStore(client=client, collection_name="test_store")

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader(
        "/Users/johnw/src/llama-index/docs",
        file_extractor=file_extractor
    ).load_data()

    # pipeline = IngestionPipeline(
    #     transformations=[
    #         SentenceSplitter(
    #             chunk_size=Settings.chunk_size,
    #             chunk_overlap=Settings.chunk_overlap
    #         ),
    #         TitleExtractor(),
    #         Settings.embed_model,
    #     ],
    #     # vector_store=vector_store,
    #     docstore=SimpleDocumentStore()
    # )

    # nodes = pipeline.run(documents=documents# , num_workers=4
    # index = VectorStoreIndex(nodes)

    # index = VectorStoreIndex.from_vector_store(
    #     vector_store,
    #     verbose=verbose
    # )
    # pipeline.persist(PERSIST_DIR)

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=Settings.embed_model,
    )
    index.storage_context.persist(persist_dir=PERSIST_DIR)

else:
    # documents = SimpleDirectoryReader(
    #     "/Users/johnw/src/llama-index/docs",
    #     file_extractor=file_extractor
    # ).load_data()
    # pipeline.load(PERSIST_DIR)
    # pipeline.run(documents=documents)
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# # configure retriever
# retriever = VectorIndexRetriever(
#     index=index,
#     similarity_top_k=10,
# )

# # configure response synthesizer
# response_synthesizer = get_response_synthesizer(
#     response_mode="tree_summarize",
# )

### Retrieving

### Prompts

qa_prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are a planner and organizer, you create agendsa for meetings where people\n"
    "can come together and share ideas, plan, and learn about how to facilitate\n"
    "larger groups. You present your information in a kindly yet efficient\n"
    "manner,and use precise language to make details very clear for those who read\n"
    "your letters.\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)

refine_prompt_str = (
    "We have the opportunity to refine the original answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "You are a planner and organizer, you create agendsa for meetings where people\n"
    "can come together and share ideas, plan, and learn about how to facilitate\n"
    "larger groups. You present your information in a kindly yet efficient\n"
    "manner,and use precise language to make details very clear for those who read\n"
    "your letters.\n"
    "Given the new context, refine the original answer to better "
    "answer the question: {query_str}. "
    "If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)

# Text QA Prompt
chat_text_qa_msgs = [
    (
        "system",
        "Always answer the question, even if the context isn't helpful.",
    ),
    ("user", qa_prompt_str),
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

# Refine Prompt
chat_refine_msgs = [
    (
        "system",
        "Always answer the question, even if the context isn't helpful.",
    ),
    ("user", refine_prompt_str),
]
refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

### Querying

# retriever = index.as_retriever()
# nodes = retriever.retrieve(    '''
# What is an effect way to approach the difficult task of mobilizing believers
# to aid in the ongoing activities of a necleus operating within a focus
# neighborhood?
#     '''
# )
# print(nodes)

def submit_query(
        query_text,
        model,
        context_window: int = 32768
):
    llm = setup_model(model)

    query_engine = index.as_query_engine(
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        llm=llm,
        streaming=True,
        similarity_top_k=8,
        verbose=verbose
    )
    # query_engine = RetrieverQueryEngine(
    #     retriever=retriever,
    #     response_synthesizer=response_synthesizer,
    #     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    # )

    return query_engine.query(query_text)

models = [
    "deepseek-r1:32b",
    "falcon3:10b",
    "phi4:latest",
    "dolphin3:latest",
    "deepseek-coder:33b",
    "gemma2:27b",
    "qwen2.5-coder:32b",
    "wizardcoder:33b",
    "codellama:70b",
    "mistral-small:latest",
    "qwen2.5:14b",
    "qwen2.5:32b",
]

for model in models:
    print("========================================================================")
    print("Model: ", model)
    print("========================================================================")
    submit_query(
        '''
What are the essential features of a nucleus, and how does it operate?
''',
        model=model
    ).print_response_stream()
