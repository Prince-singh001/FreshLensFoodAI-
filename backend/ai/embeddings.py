"""
FoodLens-AI - Vector Embeddings & Similarity Engine
Uses lightweight TF-IDF and cosine similarity for fast, memory-safe,
production-ready document retrieval on low-memory and cloud deployments.
"""
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FoodKnowledgeVectorizer:
    """Vectorizes text documents and queries for semantic and keyword retrieval."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.doc_vectors = None
        self.documents: List[Dict[str, Any]] = []

    def fit_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.documents = documents
        corpus = [f"{doc['title']} {doc['category']} {doc['content']}" for doc in documents]
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        if self.doc_vectors is None or not self.documents:
            return []

        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.05:  # Relevance threshold
                results.append((self.documents[idx], score))
        return results

# Singleton vectorizer instance
_vectorizer_instance = None

def get_food_vectorizer(docs: List[Dict[str, Any]] = None) -> FoodKnowledgeVectorizer:
    global _vectorizer_instance
    if _vectorizer_instance is None:
        _vectorizer_instance = FoodKnowledgeVectorizer()
        if docs:
            _vectorizer_instance.fit_documents(docs)
    return _vectorizer_instance
