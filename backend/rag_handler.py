"""
RAG Handler Module
Handles all RAG-related calls from backend.py using ChromaDB for vector search.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ChromaDB client cache
_chroma_client = None
_chroma_collection = None

DATA_ROOT = Path(__file__).parent / "data"
CHROMA_DB_PATH = DATA_ROOT / "rag_chunks" / "chroma_db"
COLLECTION_NAME = "cp_rag"  # Default collection name, matches RAG pipeline


def initialize_rag(
    db_path: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
    embedder: str = "local",
    local_model: str = "BAAI/bge-small-en-v1.5",
    openai_api_key: Optional[str] = None,
) -> bool:
    """
    Initialize the RAG system by connecting to ChromaDB.
    
    Args:
        db_path: Path to ChromaDB directory. Defaults to data/rag_chunks/chroma_db
        collection_name: Name of the ChromaDB collection to use
        embedder: "local" or "openai"
        local_model: Local embedding model name (used if embedder="local")
        openai_api_key: OpenAI API key (used if embedder="openai")
    
    Returns:
        True if initialization successful, False otherwise
    """
    global _chroma_client, _chroma_collection
    
    try:
        import chromadb
        from chromadb.utils import embedding_functions as ef
    except ImportError:
        logger.error("ChromaDB not installed. Run: pip install chromadb")
        return False
    
    # Use provided path or default
    db_path = db_path or str(CHROMA_DB_PATH)
    
    # Check if database exists
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        logger.warning(f"ChromaDB path not found: {db_path}")
        logger.info("Please run: python RAG/main.py --embedder local --vector-db chroma")
        return False
    
    try:
        # Create persistent client
        _chroma_client = chromadb.PersistentClient(path=str(db_path))
        
        # Set up embedding function
        if embedder == "local":
            embedding_fn = ef.SentenceTransformerEmbeddingFunction(
                model_name=local_model,
                normalize_embeddings=True,
            )
            logger.info(f"RAG initialized with local embedder: {local_model}")
        elif embedder == "openai":
            if not openai_api_key:
                logger.error("OpenAI API key required for 'openai' embedder")
                return False
            embedding_fn = ef.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small",
            )
            logger.info("RAG initialized with OpenAI embedder: text-embedding-3-small")
        else:
            logger.error(f"Unknown embedder: {embedder}")
            return False
        
        # Get or create collection
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        
        count = _chroma_collection.count()
        logger.info(f"Connected to ChromaDB collection '{collection_name}' with {count} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")
        return False


def query_rag(
    query_text: str,
    n_results: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Query the RAG system for relevant chunks.
    
    Args:
        query_text: The query string
        n_results: Number of results to return
        filters: Optional ChromaDB filters (e.g., {"source": {"$eq": "codeforces"}})
    
    Returns:
        List of retrieved chunks with metadata and similarity scores
    """
    global _chroma_collection
    
    if _chroma_collection is None:
        logger.warning("RAG not initialized. Call initialize_rag() first.")
        return []
    
    try:
        logger.info(f"🔍 Querying RAG with: '{query_text}'")
        
        # Query the collection
        results = _chroma_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filters,
            # The caller only consumes documents, metadata, and distances.
            # Returning embedding vectors adds avoidable serialization work.
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results["documents"] and len(results["documents"]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                
                # Convert distance to similarity score (1 - distance for cosine)
                similarity = 1 - distance if distance is not None else 0
                
                formatted_results.append({
                    "text": doc,
                    "source": metadata.get("source", "unknown"),
                    "chunk_type": metadata.get("chunk_type", "unknown"),
                    "title": metadata.get("title", ""),
                    "similarity": round(similarity, 3),
                    "metadata": metadata,
                })
        
        logger.info(f"✓ RAG query returned {len(formatted_results)} results")
        for i, result in enumerate(formatted_results, 1):
            logger.info(f"  [{i}] {result['source']} ({result['chunk_type']}) - Similarity: {result['similarity']} - {result['title']}")
        return formatted_results
        
    except Exception as e:
        logger.error(f"✗ RAG query failed: {e}", exc_info=True)
        return []


def format_rag_context(
    query_results: List[Dict[str, Any]],
    include_metadata: bool = True,
) -> str:
    """
    Format RAG query results into a readable context string.
    
    Args:
        query_results: List of results from query_rag()
        include_metadata: Whether to include metadata in the formatted output
    
    Returns:
        Formatted string ready to append to LLM prompt
    """
    if not query_results:
        return ""
    
    lines = ["## Context from Competitive Programming Knowledge Base:\n"]
    
    for i, result in enumerate(query_results, 1):
        lines.append(f"### Result {i} (Similarity: {result['similarity']})")
        if result["title"]:
            lines.append(f"**Title:** {result['title']}")
        if include_metadata:
            lines.append(f"**Source:** {result['source']} ({result['chunk_type']})")
        lines.append("")
        lines.append(result["text"])
        lines.append("")
    
    return "\n".join(lines)


def get_rag_augmented_response(
    query_text: str,
    system_prompt: str,
    n_results: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Get RAG context for a query and format it with the system prompt.
    
    Args:
        query_text: The user's query
        system_prompt: Original system prompt
        n_results: Number of RAG results to retrieve
        filters: Optional ChromaDB filters
        include_metadata: Whether to include metadata in formatted context
    
    Returns:
        Tuple of (augmented_system_prompt, rag_results)
    """
    rag_results = query_rag(query_text, n_results=n_results, filters=filters)
    
    rag_context = format_rag_context(rag_results, include_metadata=include_metadata)
    
    augmented_prompt = system_prompt
    if rag_context:
        augmented_prompt = f"{system_prompt}\n\n{rag_context}"
    
    return augmented_prompt, rag_results


def is_rag_available() -> bool:
    """Check if RAG system is initialized and ready to use."""
    return _chroma_collection is not None
