"""Retrieval implementations for RAG client."""

from typing import final, no_type_check, override

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore


@final
class CustomRetriever(BaseRetriever):
    """Custom retriever performing both semantic search and hybrid search."""
    
    @no_type_check
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        verbose: bool = False,
        mode: str = "AND",
    ) -> None:
        """Init params."""
        
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__(verbose=verbose)
    
    @override
    @no_type_check
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""
        
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
        
        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}
        
        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})
        
        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)
        
        return [combined_dict[rid] for rid in retrieve_ids]