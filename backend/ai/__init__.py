"""
FoodLens-AI - Agentic AI Package
RAG retrieval, modular tools, multilingual reasoning, and conversational memory.
"""
from .agent import get_agent, FoodLensAgent, FreshLensAgent
from .retriever import get_retriever, FoodKnowledgeRetriever
from .memory import get_memory, ConversationMemory
from .tools import (
    tool_food_knowledge_retriever,
    tool_storage_recommendation,
    tool_food_safety_protocol,
    tool_nutrition_lookup,
    tool_current_scan_context,
    tool_scan_history
)

__all__ = [
    "get_agent",
    "FoodLensAgent",
    "FreshLensAgent",
    "get_retriever",
    "FoodKnowledgeRetriever",
    "get_memory",
    "ConversationMemory",
    "tool_food_knowledge_retriever",
    "tool_storage_recommendation",
    "tool_food_safety_protocol",
    "tool_nutrition_lookup",
    "tool_current_scan_context",
    "tool_scan_history"
]
