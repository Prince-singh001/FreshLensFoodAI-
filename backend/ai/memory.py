"""
FreshLens AI - Conversation Memory Manager
Maintains in-memory chat turns and context window per conversation_id.
"""
import time
from typing import List, Dict, Any

class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        if conversation_id not in self.sessions:
            self.sessions[conversation_id] = []
        
        self.sessions[conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Enforce rolling window
        if len(self.sessions[conversation_id]) > self.max_turns * 2:
            self.sessions[conversation_id] = self.sessions[conversation_id][-self.max_turns * 2:]

    def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(conversation_id, [])

    def clear(self, conversation_id: str) -> None:
        if conversation_id in self.sessions:
            del self.sessions[conversation_id]

# Singleton instance
_memory_instance = None

def get_memory() -> ConversationMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance
