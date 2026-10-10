"""
FoodLens-AI - Chatbot Service Wrapper
Connects API endpoints to the Agentic AI Orchestrator with RAG retrieval,
conversation memory, and multilingual synthesis (English, Hindi, etc.).
"""
from typing import Dict, Any, Optional

try:
    from ai.agent import get_agent
except ImportError:
    from backend.ai.agent import get_agent

def process_agent_chat(
    message: str,
    conversation_id: Optional[str] = None,
    language: str = "en",
    scan_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Processes chat queries through the LangChain agentic reasoning pipeline."""
    agent = get_agent()
    return agent.process_message(
        message=message,
        conversation_id=conversation_id,
        language=language,
        scan_context=scan_context
    )

def get_chatbot_response(user_message: str) -> str:
    """Legacy compatibility function returning string answer."""
    res = process_agent_chat(user_message)
    return res.get("answer", "")
