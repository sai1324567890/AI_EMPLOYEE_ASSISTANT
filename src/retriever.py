
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .document_loader import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("Cannot build a retriever from an empty corpus.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        corpus_texts = [f"{c.doc_title} {c.section} {c.text}" for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus_texts)

        # Pre-compute index positions per route for fast slicing
        self.route_indices: Dict[str, List[int]] = {}
        for i, c in enumerate(chunks):
            self.route_indices.setdefault(c.route, []).append(i)

    def _query_vec(self, query: str):
        return self.vectorizer.transform([query])

    def score_routes(self, query: str) -> Dict[str, float]:
        """
        Returns the best cosine-similarity score achieved by `query`
        against each route's chunks. Used by the router to decide
        which knowledge source (if any) is confidently relevant.
        """
        qvec = self._query_vec(query)
        sims = cosine_similarity(qvec, self.matrix)[0]
        scores = {}
        for route, idxs in self.route_indices.items():
            if not idxs:
                scores[route] = 0.0
                continue
            scores[route] = float(np.max(sims[idxs]))
        return scores

    def top_k(self, query: str, route: str, k: int = 3) -> List[RetrievedChunk]:
        idxs = self.route_indices.get(route, [])
        if not idxs:
            return []
        qvec = self._query_vec(query)
        sims = cosine_similarity(qvec, self.matrix[idxs])[0]
        order = np.argsort(sims)[::-1][:k]
        results = []
        for o in order:
            global_idx = idxs[o]
            score = float(sims[o])
            # Same floor used for routing confidence — an individually weak
            # chunk shouldn't be treated as "found" just because a handful
            # of stopword-level tokens overlap.
            if score < config.ROUTING_CONFIDENCE_THRESHOLD:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[global_idx], score=score))
        return results
