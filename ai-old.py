#!./.venv/bin/python

# python -m venv .venv
# source .venv/bin/activate
# pip install llama-index "numpy<2" chromadb llama-index-vector-stores-chroma pypdf llama-index-llms-ollama llama-index-embeddings-huggingface
# python starter.py

import os
import chromadb

from llama_index.cli.rag import RagCLI, default_ragcli_persist_dir
from llama_index.cli.rag.base import QueryPipelineQueryEngine, query_input
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.query_pipeline.components.function import FnComponent
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="deepseek-r1:8b", request_timeout=360.0, temperature=0.6)
Settings.llm = llm

persist_dir = default_ragcli_persist_dir()
chroma_client = chromadb.PersistentClient(path=persist_dir)
chroma_collection = chroma_client.create_collection("default", get_or_create=True)
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection, persist_dir=persist_dir
)
# docstore = SimpleDocumentStore()

ingestion_pipeline = IngestionPipeline(
    transformations=[],
    vector_store=vector_store,
    # docstore=docstore,
    # cache=IngestionCache(),
)

verbose = False

query_component = FnComponent(
    fn=query_input, output_key="output", req_params={"query_str"}
)
retriever = VectorStoreIndex.from_vector_store(
    ingestion_pipeline.vector_store,
).as_retriever(similarity_top_k=8)
response_synthesizer = CompactAndRefine(streaming=True, verbose=verbose)

query_pipeline = QueryPipeline(verbose=verbose)
query_pipeline.add_modules(
    {
        "query": query_component,
        "retriever": retriever,
        "summarizer": response_synthesizer,
    }
)
query_pipeline.add_link("query", "retriever")
query_pipeline.add_link("retriever", "summarizer", dest_key="nodes")
query_pipeline.add_link("query", "summarizer", dest_key="query_str")

query_engine = QueryPipelineQueryEngine(query_pipeline=query_pipeline)
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine, llm=llm, verbose=verbose
)

# file_extractor = {".html": ...}

rag_cli_instance = RagCLI(
    ingestion_pipeline=ingestion_pipeline,
    llm=llm,
    query_pipeline=query_pipeline,
    chat_engine=chat_engine,
    # file_extractor=file_extractor,
)

if __name__ == "__main__":
    rag_cli_instance.cli()
