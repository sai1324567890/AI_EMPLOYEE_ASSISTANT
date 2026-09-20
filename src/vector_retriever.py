from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .document_loader import Chunk
from . import config


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def _try_load_sentence_transformer():
    """Best-effort load of a local sentence-transformers model.
    Returns None if the package/model isn't available - the vector
    retriever then falls back to LSA embeddings automatically."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        import os

        # Only attempt this if the user has explicitly pointed at a
        # local model directory - we never trigger a network download
        # from inside this project.
        local_path = os.getenv("SENTENCE_TRANSFORMERS_LOCAL_PATH", "")
        if not local_path:
            return None
        model = SentenceTransformer(local_path)
        return model
    except Exception:
        return None


class VectorRetriever:
    """FAISS-backed dense retriever. Same public surface as Retriever."""

    def __init__(self, chunks: List[Chunk], embedding_dim: int = 128):
        if not chunks:
            raise ValueError("Cannot build a vector retriever from an empty corpus.")
        self.chunks = chunks
        self.embedding_dim = embedding_dim

        texts = [f"{c.doc_title} {c.section} {c.text}" for c in chunks]

        self._st_model = _try_load_sentence_transformer()
        if self._st_model is not None:
            self.embedding_backend = "sentence-transformers"
            vectors = self._st_model.encode(texts, normalize_embeddings=False)
            vectors = np.asarray(vectors, dtype="float32")
        else:
            self.embedding_backend = "tfidf-svd (LSA)"
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            n_components = min(embedding_dim, min(tfidf_matrix.shape) - 1)
            n_components = max(n_components, 2)
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            vectors = self.svd.fit_transform(tfidf_matrix).astype("float32")

        vectors = self._normalize(vectors)
        self.vectors = vectors
        self.dim = vectors.shape[1]

        # Global index (used for routing scores across all chunks)
        self.global_index = faiss.IndexFlatIP(self.dim)
        self.global_index.add(vectors)

        # Per-route index for fast top-k retrieval once a route is chosen
        self.route_indices: Dict[str, List[int]] = {}
        for i, c in enumerate(chunks):
            self.route_indices.setdefault(c.route, []).append(i)

        self._route_faiss_index: Dict[str, faiss.IndexFlatIP] = {}
        for route, idxs in self.route_indices.items():
            sub_vectors = vectors[idxs]
            idx = faiss.IndexFlatIP(self.dim)
            if len(idxs):
                idx.add(sub_vectors)
            self._route_faiss_index[route] = idx

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype("float32")

    def _embed_query(self, query: str) -> np.ndarray:
        if self._st_model is not None:
            vec = self._st_model.encode([query], normalize_embeddings=False)
            vec = np.asarray(vec, dtype="float32")
        else:
            tfidf_vec = self.vectorizer.transform([query])
            vec = self.svd.transform(tfidf_vec).astype("float32")
        return self._normalize(vec)

    def score_routes(self, query: str) -> Dict[str, float]:
        qvec = self._embed_query(query)
        scores = {}
        for route, idx in self._route_faiss_index.items():
            if idx.ntotal == 0:
                scores[route] = 0.0
                continue
            k = min(1, idx.ntotal)
            sims, _ = idx.search(qvec, k)
            scores[route] = float(max(sims[0][0], 0.0))
        return scores

    def top_k(self, query: str, route: str, k: int = 3) -> List[RetrievedChunk]:
        idx = self._route_faiss_index.get(route)
        idxs = self.route_indices.get(route, [])
        if idx is None or not idxs:
            return []
        qvec = self._embed_query(query)
        k = min(k, idx.ntotal)
        sims, positions = idx.search(qvec, k)
        results = []
        for score, pos in zip(sims[0], positions[0]):
            if pos < 0 or score < config.ROUTING_CONFIDENCE_THRESHOLD:
                continue
            global_idx = idxs[pos]
            results.append(RetrievedChunk(chunk=self.chunks[global_idx], score=float(score)))
        return results
