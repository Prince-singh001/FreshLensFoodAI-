"""
FoodLens-AI - RAG Document Retriever
Retrieves grounded food safety documents and formats citation metadata.
"""
from typing import List, Dict, Any, Tuple
from .rag import get_knowledge_corpus
from .embeddings import get_food_vectorizer

class FoodKnowledgeRetriever:
    """Retriever for querying verified food-safety and produce guidelines."""
    def __init__(self):
        docs = get_knowledge_corpus()
        self.vectorizer = get_food_vectorizer(docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        matches = self.vectorizer.query(query, top_k=top_k)
        retrieved_items = []
        for doc, score in matches:
            retrieved_items.append({
                "id": doc.get("id"),
                "title": doc.get("title"),
                "source": doc.get("source"),
                "url": doc.get("url"),
                "category": doc.get("category"),
                "content": doc.get("content"),
                "score": round(score, 4)
            })
        return retrieved_items

    def retrieve_formatted_context(self, query: str, top_k: int = 2) -> Tuple[str, List[Dict[str, str]]]:
        """Returns formatted string context for LLM prompt and list of source citations."""
        items = self.retrieve(query, top_k=top_k)
        if not items:
            return "", []

        context_lines = []
        sources = []
        for idx, item in enumerate(items, 1):
            context_lines.append(f"[{idx}] Source: {item['source']} - '{item['title']}'\n{item['content']}")
            sources.append({
                "title": item["title"],
                "source": item["source"],
                "url": item["url"],
                "category": item["category"]
            })

        return "\n\n".join(context_lines), sources

# Singleton instance
_retriever_instance = None

def get_retriever() -> FoodKnowledgeRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = FoodKnowledgeRetriever()
    return _retriever_instance
