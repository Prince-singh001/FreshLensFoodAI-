"""
FreshLens AI Chatbot Compatibility Wrapper
"""
try:
    from services.chatbot_service import get_chatbot_response
except ImportError:
    from backend.services.chatbot_service import get_chatbot_response

__all__ = ["get_chatbot_response"]
