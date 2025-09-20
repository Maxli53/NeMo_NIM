"""
Local Embeddings Manager
Handles embedding generation using Sentence Transformers or other local models
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional, Union, Dict
import torch
import logging
from dataclasses import dataclass
from src.config import config, EmbeddingProvider
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    embeddings: np.ndarray
    model_name: str
    dimension: int
    texts: List[str]


class LocalEmbeddingManager:
    """Manages local embedding models for RAG"""

    def __init__(self, model_name: Optional[str] = None):
        self.provider = config.embedding_config.provider
        self.model_name = model_name or config.embedding_config.model
        self.device = config.embedding_config.device
        self.batch_size = config.embedding_config.batch_size
        self.model = None
        self.dimension = None
        self._embedding_cache = {}  # Cache for embeddings

        if self.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            self._load_sentence_transformer()

    def _load_sentence_transformer(self):
        """Load Sentence Transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            # Get embedding dimension
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.dimension = len(test_embedding)
            config.embedding_config.dimension = self.dimension
            logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """Generate embeddings for text"""

        if self.model is None:
            raise RuntimeError("Embedding model not loaded")

        # Handle single text
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False

        # Check cache if enabled
        if config.cache_embeddings:
            cached_embeddings = []
            texts_to_embed = []
            for t in texts:
                if t in self._embedding_cache:
                    cached_embeddings.append(self._embedding_cache[t])
                else:
                    texts_to_embed.append(t)

            # If all cached, return immediately
            if not texts_to_embed:
                embeddings = np.array(cached_embeddings)
                return embeddings[0] if single_input else embeddings

            # Generate embeddings for non-cached texts
            if texts_to_embed:
                new_embeddings = self.model.encode(
                    texts_to_embed,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True
                )

                # Cache new embeddings
                for t, emb in zip(texts_to_embed, new_embeddings):
                    self._embedding_cache[t] = emb

                # Combine cached and new embeddings
                all_embeddings = []
                new_idx = 0
                for t in texts:
                    if t in cached_embeddings:
                        all_embeddings.append(self._embedding_cache[t])
                    else:
                        all_embeddings.append(new_embeddings[new_idx])
                        new_idx += 1

                embeddings = np.array(all_embeddings)
        else:
            # No caching
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )

        return embeddings[0] if single_input else embeddings

    async def embed_text_async(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """Async wrapper for embedding generation"""
        return await asyncio.to_thread(
            self.embed_text, text, normalize
        )

    def embed_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True
    ) -> EmbeddingResult:
        """Embed a batch of texts and return structured result"""
        embeddings = self.embed_text(texts, normalize, show_progress)
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            dimension=self.dimension,
            texts=texts
        )

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """Compute similarity between query and documents"""

        if metric == "cosine":
            # Normalize if not already done
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = document_embeddings / np.linalg.norm(
                document_embeddings, axis=1, keepdims=True
            )
            similarities = np.dot(doc_norms, query_norm)
        elif metric == "euclidean":
            similarities = -np.linalg.norm(
                document_embeddings - query_embedding, axis=1
            )
        elif metric == "dot":
            similarities = np.dot(document_embeddings, query_embedding)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

        return similarities

    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """Rerank documents based on similarity to query"""

        # Generate embeddings
        query_embedding = self.embed_text(query, normalize=True)
        doc_embeddings = self.embed_text(documents, normalize=True)

        # Compute similarities
        similarities = self.compute_similarity(query_embedding, doc_embeddings)

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        # Return top-k if specified
        if top_k:
            sorted_indices = sorted_indices[:top_k]

        results = []
        for idx in sorted_indices:
            results.append({
                "text": documents[idx],
                "score": float(similarities[idx]),
                "rank": len(results) + 1
            })

        return results

    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the embedding model"""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "batch_size": self.batch_size,
            "cache_size": len(self._embedding_cache)
        }


class MultiModalEmbeddingManager(LocalEmbeddingManager):
    """Extended manager for multi-modal embeddings (text + images)"""

    def embed_image(self, image_path: str) -> np.ndarray:
        """Generate embeddings for images (if model supports it)"""
        # This would be implemented for multi-modal models
        raise NotImplementedError("Image embeddings not yet supported")

    def embed_document(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> np.ndarray:
        """Embed document with metadata awareness"""
        # Could incorporate metadata into embeddings
        if metadata:
            # Augment text with metadata
            augmented_text = f"{text}\n[Metadata: {metadata}]"
            return self.embed_text(augmented_text)
        return self.embed_text(text)


# Global embedding manager instance
embedding_manager = LocalEmbeddingManager()