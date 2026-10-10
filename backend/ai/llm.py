"""
FoodLens-AI - LLM Client Interface
Configures LangChain chat models if an API key is provided,
or provides a deterministic, grounded reasoning engine as fallback.
"""
import os
from typing import Dict, Any, Optional

class LLMClient:
    """Manages LLM communication via LangChain or grounded fallback."""
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY")
        self.chat_model = None

        if self.api_key:
            try:
                # Try LangChain ChatOpenAI if configured
                from langchain_community.chat_models import ChatOpenAI
                base_url = os.environ.get("LLM_API_URL")
                kwargs = {"api_key": self.api_key, "temperature": 0.2}
                if base_url:
                    kwargs["base_url"] = base_url
                self.chat_model = ChatOpenAI(**kwargs)
            except Exception:
                self.chat_model = None

    def is_cloud_enabled(self) -> bool:
        return self.chat_model is not None

# Singleton instance
_llm_client = None

def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
