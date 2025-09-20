"""
Enhanced FAISS Vector Database with Local Embeddings
Supports both API and local embedding models
"""

import numpy as np
import faiss
from typing import List, Dict, Optional, Any, Tuple
from openai import OpenAI
import asyncio
from src.config import config, EmbeddingProvider
from src.utils.embeddings import embedding_manager, EmbeddingResult
import pickle
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FAISSVectorDB:
    """Enhanced FAISS vector database with local embedding support"""

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        index_type: str = "flat",
        use_gpu: bool = False
    ):
        self.embedding_dim = embedding_dim or config.embedding_config.dimension
        self.index_type = index_type
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.index = None
        self.entries = []
        self.metadata = {}
        self._init_index()

        # For API embeddings (fallback)
        self.openai_client = OpenAI(api_key=config.openai_api_key) if config.openai_api_key else None

    def _init_index(self):
        """Initialize FAISS index based on type"""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Move to GPU if available and requested
        if self.use_gpu:
            try:
                import faiss.contrib.torch_utils
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
                self.use_gpu = False

    def add(
        self,
        embedding: np.ndarray,
        text: str,
        source: str,
        metadata: Optional[Dict] = None
    ):
        """Add a single entry to the database"""
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {embedding.shape[0]} != {self.embedding_dim}")

        self.index.add(np.array([embedding], dtype='float32'))
        entry = {
            "text": text,
            "source": source,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        self.entries.append(entry)

    def add_batch(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        sources: List[str],
        metadata_list: Optional[List[Dict]] = None
    ):
        """Add multiple entries at once"""
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.embedding_dim}")

        embeddings_array = np.array(embeddings, dtype='float32')
        self.index.add(embeddings_array)

        for i, (text, source) in enumerate(zip(texts, sources)):
            metadata = metadata_list[i] if metadata_list else {}
            entry = {
                "text": text,
                "source": source,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
            self.entries.append(entry)

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """Search for similar entries"""
        if self.index.ntotal == 0:
            return []

        query_array = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))

        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.entries):
                # Apply threshold if specified
                if threshold and dist > threshold:
                    continue

                entry = self.entries[idx].copy()
                entry.update({
                    "distance": float(dist),
                    "rank": i + 1,
                    "score": 1.0 / (1.0 + float(dist))  # Convert distance to similarity score
                })
                results.append(entry)

        return results

    def embed_and_search(
        self,
        query_text: str,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """Embed query text and perform search"""
        if config.embedding_config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            query_embedding = embedding_manager.embed_text(query_text)
        else:
            query_embedding = self._embed_with_api(query_text)

        return self.similarity_search(query_embedding, top_k, threshold)

    async def embed_and_search_async(
        self,
        query_text: str,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """Async version of embed_and_search"""
        query_embedding = await embedding_manager.embed_text_async(query_text)
        return self.similarity_search(query_embedding, top_k, threshold)

    def _embed_with_api(self, text: str) -> np.ndarray:
        """Fallback to OpenAI API for embeddings"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not configured")

        try:
            response = self.openai_client.embeddings.create(
                model=config.embedding_config.api_model,
                input=text
            )
            return np.array(response.data[0].embedding, dtype='float32')
        except Exception as e:
            logger.error(f"API embedding failed: {e}")
            raise

    def clear(self):
        """Clear all entries from the database"""
        self._init_index()
        self.entries = []
        self.metadata = {}

    @property
    def size(self) -> int:
        """Get the number of entries in the database"""
        return self.index.ntotal

    def save(self, filepath: str):
        """Save the index and entries to disk"""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")

        # Save entries and metadata
        with open(f"{filepath}.data", 'wb') as f:
            pickle.dump({
                'entries': self.entries,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f)
        logger.info(f"Vector DB saved to {filepath}")

    def load(self, filepath: str):
        """Load the index and entries from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")

        # Load entries and metadata
        with open(f"{filepath}.data", 'rb') as f:
            data = pickle.load(f)
            self.entries = data['entries']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
        logger.info(f"Vector DB loaded from {filepath}")

    def update_entry(self, idx: int, text: Optional[str] = None, metadata: Optional[Dict] = None):
        """Update an existing entry's text or metadata"""
        if idx >= len(self.entries):
            raise IndexError(f"Entry {idx} does not exist")

        if text is not None:
            self.entries[idx]['text'] = text
        if metadata is not None:
            self.entries[idx]['metadata'].update(metadata)

    def remove_entry(self, idx: int):
        """Remove an entry (note: FAISS doesn't support removal, so this marks as deleted)"""
        if idx >= len(self.entries):
            raise IndexError(f"Entry {idx} does not exist")

        self.entries[idx]['deleted'] = True

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        return {
            "total_entries": self.size,
            "embedding_dimension": self.embedding_dim,
            "index_type": self.index_type,
            "use_gpu": self.use_gpu,
            "unique_sources": len(set(e['source'] for e in self.entries if not e.get('deleted'))),
            "deleted_entries": sum(1 for e in self.entries if e.get('deleted'))
        }


class VectorDBManager:
    """Manages multiple vector databases for different domains"""

    def __init__(self):
        self.databases = {}
        self.embedding_dim = config.embedding_config.dimension

    def get_or_create_db(
        self,
        domain: str,
        index_type: str = "flat",
        use_gpu: bool = False
    ) -> FAISSVectorDB:
        """Get existing or create new database for a domain"""
        if domain not in self.databases:
            self.databases[domain] = FAISSVectorDB(
                embedding_dim=self.embedding_dim,
                index_type=index_type,
                use_gpu=use_gpu
            )
            # Try to load from disk if exists
            db_path = os.path.join(config.index_dir, f"{domain}_db")
            if os.path.exists(f"{db_path}.index"):
                try:
                    self.databases[domain].load(db_path)
                    logger.info(f"Loaded existing database for domain: {domain}")
                except Exception as e:
                    logger.warning(f"Failed to load database for {domain}: {e}")

        return self.databases[domain]

    def get_db(self, domain: str) -> Optional[FAISSVectorDB]:
        """Get database for a domain if it exists"""
        return self.databases.get(domain)

    def list_databases(self) -> List[str]:
        """List all available databases"""
        return list(self.databases.keys())

    def save_all(self):
        """Save all databases to disk"""
        for domain, db in self.databases.items():
            db_path = os.path.join(config.index_dir, f"{domain}_db")
            try:
                db.save(db_path)
            except Exception as e:
                logger.error(f"Failed to save database {domain}: {e}")

    def clear_all(self):
        """Clear all databases"""
        for db in self.databases.values():
            db.clear()
        self.databases = {}

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all databases"""
        stats = {}
        for domain, db in self.databases.items():
            stats[domain] = db.get_statistics()
        return stats


# Global instance
vector_db_manager = VectorDBManager()