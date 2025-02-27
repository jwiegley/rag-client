#!./.venv/bin/python

# python -m venv .venv
# source .venv/bin/activate
# pip install llama-index "numpy<2" chromadb llama-index-vector-stores-chroma pypdf llama-index-llms-ollama llama-index-embeddings-huggingface
# python starter.py

import os
import chromadb
import re

from llama_index.cli.rag import RagCLI, default_ragcli_persist_dir
from llama_index.cli.rag.base import QueryPipelineQueryEngine, query_input
from llama_index.core import ChatPromptTemplate, PromptTemplate
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.schema import TransformComponent
from llama_index.core.query_pipeline.components.function import FnComponent
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2:1b", request_timeout=360.0, temperature=1.0)
Settings.llm = llm

persist_dir = default_ragcli_persist_dir()
print('persist_dir =', persist_dir)

chroma_client = chromadb.PersistentClient(path=persist_dir)
chroma_collection = chroma_client.create_collection("default", get_or_create=True)
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection, persist_dir=persist_dir
)
docstore = SimpleDocumentStore()

class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
        return nodes

ingestion_pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=512, chunk_overlap=128),
        TitleExtractor(nodes=5),
        TextCleaner(),
        HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5"),
        QuestionsAnsweredExtractor(questions=3),
    ],
    vector_store=vector_store,
    docstore=docstore,
    cache=IngestionCache(),
)

documents = SimpleDirectoryReader(
    "/Users/johnw/kadena/kadena-docs/docs/pact-5/capabilities/",
    recursive=True
).load_data()

nodes = ingestion_pipeline.run(documents=documents)

verbose = True

query_component = FnComponent(
    fn=query_input, output_key="output", req_params={"query_str"}
)
retriever = VectorStoreIndex.from_vector_store(
    ingestion_pipeline.vector_store,
).as_retriever(similarity_top_k=8)

basic_prompt = "Please answer my questions in a succinct and helpful way."

qa_prompt_str = '''
Context information is below.
---------------------
{context_str}
---------------------
''' + basic_prompt + '''
Given the context information and not prior knowledge, answer the question:
{query_str}
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
    (
        "user",
        qa_prompt_str
    ),
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

chat_refine_msgs = [
    (
        "system",
        "Always answer the question, even if the context isn't helpful.",
    ),
    (
        "user",
        refine_prompt_str
    ),
]
refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

response_synthesizer = CompactAndRefine(
    text_qa_template=text_qa_template,
    refine_template=refine_template,
    llm=llm,
    streaming=True,
    verbose=verbose)

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
