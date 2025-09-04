#!/usr/bin/env python3
"""Basic document indexing example.

This script demonstrates how to:
1. Load configuration
2. Initialize RAG workflow
3. Index documents
4. Perform queries
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_client.core.workflow import RAGWorkflow
from rag_client.config.loader import load_config
from rag_client.utils.logging import setup_logging


def main():
    """Main function demonstrating document indexing."""
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Load configuration
    config_path = Path(__file__).parent / "configs" / "basic.yaml"
    config = load_config(str(config_path))
    
    # Initialize workflow
    print("Initializing RAG workflow...")
    workflow = RAGWorkflow(config)
    
    # Example documents directory (adjust to your needs)
    docs_dir = Path(__file__).parent / "sample_docs"
    
    if not docs_dir.exists():
        print(f"Creating sample documents directory: {docs_dir}")
        docs_dir.mkdir(exist_ok=True)
        
        # Create sample documents
        sample_doc1 = docs_dir / "introduction.txt"
        sample_doc1.write_text("""
        Introduction to RAG (Retrieval-Augmented Generation)
        
        RAG is a technique that enhances Large Language Models (LLMs) by providing 
        them with relevant context from a knowledge base. This approach combines 
        the power of retrieval systems with generative AI models.
        
        Key benefits of RAG:
        - Reduces hallucinations by grounding responses in actual data
        - Enables up-to-date information without retraining
        - Provides source attribution for generated content
        - Allows domain-specific knowledge integration
        """)
        
        sample_doc2 = docs_dir / "implementation.md"
        sample_doc2.write_text("""
        # Implementing RAG Systems
        
        ## Components
        
        1. **Document Store**: Where your knowledge base is stored
        2. **Embedding Model**: Converts text to vector representations
        3. **Vector Database**: Stores and searches embeddings efficiently
        4. **Retriever**: Finds relevant documents for queries
        5. **LLM**: Generates responses using retrieved context
        
        ## Best Practices
        
        - Choose appropriate chunk sizes for your documents
        - Use high-quality embedding models
        - Implement hybrid search (dense + sparse)
        - Monitor and evaluate retrieval quality
        - Fine-tune prompts for your use case
        """)
    
    # Index documents
    print(f"\nIndexing documents from: {docs_dir}")
    workflow.index_documents([str(docs_dir)])
    print("Indexing complete!")
    
    # Example queries
    queries = [
        "What is RAG?",
        "What are the main components of a RAG system?",
        "How does RAG reduce hallucinations?",
        "What are best practices for implementing RAG?"
    ]
    
    print("\n" + "="*50)
    print("Testing queries:")
    print("="*50)
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        response = workflow.query(query)
        print(f"Response: {response}")
        
        # Also show sources if available
        if hasattr(response, 'source_nodes'):
            print("\nSources:")
            for node in response.source_nodes[:2]:  # Show top 2 sources
                print(f"  - Score: {node.score:.3f}")
                print(f"    Text: {node.text[:100]}...")
    
    print("\n" + "="*50)
    print("Example complete!")


if __name__ == "__main__":
    main()