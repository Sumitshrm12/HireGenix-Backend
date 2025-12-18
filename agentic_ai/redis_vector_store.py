"""
Redis Vector Store for Agentic AI
==================================

High-performance vector storage using Redis with RediSearch module.
Supports semantic search, embeddings storage, and RAG operations.
"""

import redis
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from redis.commands.search.field import TextField, VectorField, NumericField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import hashlib
from datetime import datetime


class RedisVectorStore:
    """
    Redis-based vector store with semantic search capabilities.
    
    Features:
    - Vector similarity search using HNSW algorithm
    - Metadata filtering
    - Hybrid search (vector + text)
    - Efficient batch operations
    - Automatic index management
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        index_name: str = "agentic_ai_vectors",
        vector_dim: int = 1536,  # text-embedding-3-large dimension
        distance_metric: str = "COSINE"  # COSINE, L2, IP
    ):
        """Initialize Redis vector store"""
        
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=False  # We'll handle encoding manually
        )
        
        self.index_name = index_name
        self.vector_dim = vector_dim
        self.distance_metric = distance_metric
        
        # Test connection
        try:
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as e:
            print(f"❌ Failed to connect to Redis: {e}")
            raise
        
        # Create index if it doesn't exist
        self._create_index_if_not_exists()
    
    def _create_index_if_not_exists(self):
        """Create Redis search index for vectors if it doesn't exist"""
        try:
            # Check if index exists
            self.redis_client.ft(self.index_name).info()
            print(f"✅ Index '{self.index_name}' already exists")
        except:
            # Create index
            schema = (
                VectorField(
                    "embedding",
                    "HNSW",  # Hierarchical Navigable Small World algorithm
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": self.distance_metric,
                        "INITIAL_CAP": 10000,
                    }
                ),
                TextField("text"),
                TextField("metadata"),
                TagField("doc_type"),
                TagField("source"),
                NumericField("timestamp"),
                NumericField("score"),
            )
            
            definition = IndexDefinition(
                prefix=[f"{self.index_name}:"],
                index_type=IndexType.HASH
            )
            
            self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition
            )
            print(f"✅ Created index '{self.index_name}'")
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add documents with embeddings to the vector store
        
        Args:
            documents: List of documents with 'text' field
            embeddings: List of embedding vectors
            metadata: Optional metadata for each document
            
        Returns:
            List of document IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        if metadata and len(metadata) != len(documents):
            raise ValueError("Number of metadata entries must match documents")
        
        doc_ids = []
        pipeline = self.redis_client.pipeline()
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate document ID
            doc_id = self._generate_doc_id(doc.get('text', ''), i)
            doc_key = f"{self.index_name}:{doc_id}"
            
            # Prepare document data
            doc_metadata = metadata[i] if metadata else {}
            
            # Convert embedding to bytes
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            
            # Store document
            pipeline.hset(
                doc_key,
                mapping={
                    "embedding": embedding_bytes,
                    "text": doc.get('text', ''),
                    "metadata": json.dumps(doc_metadata),
                    "doc_type": doc_metadata.get('type', 'document'),
                    "source": doc_metadata.get('source', 'unknown'),
                    "timestamp": int(datetime.utcnow().timestamp()),
                    "score": doc_metadata.get('score', 0.0),
                }
            )
            
            doc_ids.append(doc_id)
        
        # Execute pipeline
        pipeline.execute()
        print(f"✅ Added {len(doc_ids)} documents to Redis vector store")
        
        return doc_ids
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filters: Optional filters (e.g., {'doc_type': 'resume', 'source': 'linkedin'})
            score_threshold: Minimum similarity score
            
        Returns:
            List of documents with scores
        """
        # Convert query embedding to bytes
        query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
        
        # Build query
        base_query = f"*=>[KNN {k} @embedding $vec AS score]"
        
        # Add filters
        if filters:
            filter_parts = []
            for key, value in filters.items():
                if isinstance(value, list):
                    filter_parts.append(f"@{key}:{{{' | '.join(value)}}}")
                else:
                    filter_parts.append(f"@{key}:{{{value}}}")
            if filter_parts:
                base_query = f"({' '.join(filter_parts)})=>[KNN {k} @embedding $vec AS score]"
        
        query = (
            Query(base_query)
            .return_fields("text", "metadata", "doc_type", "source", "score")
            .sort_by("score")
            .dialect(2)
        )
        
        # Execute search
        results = self.redis_client.ft(self.index_name).search(
            query,
            query_params={"vec": query_bytes}
        )
        
        # Parse results
        documents = []
        for doc in results.docs:
            score = float(doc.score)
            
            # Apply score threshold if specified
            if score_threshold and score < score_threshold:
                continue
            
            documents.append({
                "id": doc.id,
                "text": doc.text,
                "metadata": json.loads(doc.metadata) if hasattr(doc, 'metadata') else {},
                "doc_type": doc.doc_type if hasattr(doc, 'doc_type') else 'unknown',
                "source": doc.source if hasattr(doc, 'source') else 'unknown',
                "similarity_score": 1.0 - score if self.distance_metric == "COSINE" else score
            })
        
        return documents
    
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        k: int = 10,
        text_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text and vector search
        
        Args:
            query_text: Text query
            query_embedding: Vector query
            k: Number of results
            text_weight: Weight for text search (0-1)
            vector_weight: Weight for vector search (0-1)
            
        Returns:
            Combined search results
        """
        # Vector search
        vector_results = self.similarity_search(query_embedding, k=k*2)
        
        # Text search
        text_query = Query(query_text).return_fields("text", "metadata", "score")
        text_results = self.redis_client.ft(self.index_name).search(text_query)
        
        # Combine results with weighted scores
        combined_scores = {}
        
        for doc in vector_results:
            doc_id = doc['id']
            combined_scores[doc_id] = {
                'doc': doc,
                'score': doc['similarity_score'] * vector_weight
            }
        
        for doc in text_results.docs:
            doc_id = doc.id
            text_score = float(doc.score) * text_weight
            
            if doc_id in combined_scores:
                combined_scores[doc_id]['score'] += text_score
            else:
                combined_scores[doc_id] = {
                    'doc': {
                        'id': doc_id,
                        'text': doc.text,
                        'metadata': json.loads(doc.metadata) if hasattr(doc, 'metadata') else {}
                    },
                    'score': text_score
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:k]
        
        return [r['doc'] for r in sorted_results]
    
    def delete_documents(self, doc_ids: List[str]) -> int:
        """Delete documents by IDs"""
        pipeline = self.redis_client.pipeline()
        for doc_id in doc_ids:
            doc_key = f"{self.index_name}:{doc_id}"
            pipeline.delete(doc_key)
        results = pipeline.execute()
        deleted_count = sum(results)
        print(f"✅ Deleted {deleted_count} documents")
        return deleted_count
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        doc_key = f"{self.index_name}:{doc_id}"
        doc_data = self.redis_client.hgetall(doc_key)
        
        if not doc_data:
            return None
        
        return {
            "id": doc_id,
            "text": doc_data.get(b'text', b'').decode('utf-8'),
            "metadata": json.loads(doc_data.get(b'metadata', b'{}').decode('utf-8')),
            "doc_type": doc_data.get(b'doc_type', b'').decode('utf-8'),
            "source": doc_data.get(b'source', b'').decode('utf-8'),
        }
    
    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the store"""
        if filters:
            filter_parts = []
            for key, value in filters.items():
                filter_parts.append(f"@{key}:{{{value}}}")
            query_str = ' '.join(filter_parts)
        else:
            query_str = "*"
        
        query = Query(query_str).no_content()
        results = self.redis_client.ft(self.index_name).search(query)
        return results.total
    
    def clear_index(self):
        """Clear all documents from the index"""
        try:
            self.redis_client.ft(self.index_name).dropindex(delete_documents=True)
            print(f"✅ Cleared index '{self.index_name}'")
            # Recreate index
            self._create_index_if_not_exists()
        except Exception as e:
            print(f"❌ Error clearing index: {e}")
    
    def _generate_doc_id(self, text: str, index: int) -> str:
        """Generate unique document ID"""
        content = f"{text}_{index}_{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def close(self):
        """Close Redis connection"""
        self.redis_client.close()
        print("✅ Redis connection closed")


# Singleton instance
_redis_store: Optional[RedisVectorStore] = None


def get_redis_vector_store(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_password: Optional[str] = None
) -> RedisVectorStore:
    """Get or create Redis vector store instance"""
    global _redis_store
    if _redis_store is None:
        _redis_store = RedisVectorStore(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_password
        )
    return _redis_store