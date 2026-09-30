"""
FreshLens AI - Modular Agent Tools
Deterministic, verifiable tools for food safety, storage recommendations,
nutrition analysis, scan context inspection, and RAG document retrieval.
"""
import os
import requests
from typing import Dict, Any, List, Optional
from .retriever import get_retriever

try:
    from ml.loader import get_classes_metadata
    from services.history_service import history_service
except ImportError:
    from backend.ml.loader import get_classes_metadata
    from backend.services.history_service import history_service

# Tool 1: RAG Food Knowledge Retriever
def tool_food_knowledge_retriever(query: str) -> Dict[str, Any]:
    """Retrieves grounded scientific food safety and produce science documentation."""
    retriever = get_retriever()
    context_text, sources = retriever.retrieve_formatted_context(query, top_k=2)
    return {
        "tool": "food_knowledge_retriever",
        "found": bool(context_text),
        "content": context_text,
        "sources": sources
    }

# Tool 2: Storage Recommendation Tool
def tool_storage_recommendation(food_name: str) -> Dict[str, Any]:
    """Returns specific storage temperatures, moisture controls, and methods."""
    metadata = get_classes_metadata().get("classes", {})
    food_clean = food_name.lower().strip()
    
    for key, data in metadata.items():
        aliases = data.get("detector_aliases", []) + [key, data.get("display_name", "").lower()]
        if any(alias in food_clean or food_clean in alias for alias in aliases):
            return {
                "tool": "storage_recommendation",
                "food": data.get("display_name", key.title()),
                "category": data.get("category"),
                "storage_tip": data.get("storage_tip"),
                "found": True
            }
            
    # Generic RAG fallback if not in starter classes
    rag_result = tool_food_knowledge_retriever(f"{food_name} storage refrigeration")
    return {
        "tool": "storage_recommendation",
        "food": food_name.title(),
        "storage_tip": rag_result.get("content") or "Store in a clean, refrigerated environment below 40°F (4°C) to extend freshness.",
        "found": rag_result.get("found", False),
        "sources": rag_result.get("sources", [])
    }

# Tool 3: Food Safety Protocol Tool
def tool_food_safety_protocol(food_name: str, condition: Optional[str] = None) -> Dict[str, Any]:
    """Evaluates danger zone, mold penetration, and physical consumption risks."""
    metadata = get_classes_metadata().get("classes", {})
    food_clean = food_name.lower().strip()
    
    matched_guideline = None
    display_name = food_name.title()
    for key, data in metadata.items():
        aliases = data.get("detector_aliases", []) + [key, data.get("display_name", "").lower()]
        if any(alias in food_clean or food_clean in alias for alias in aliases):
            matched_guideline = data.get("safety_guideline")
            display_name = data.get("display_name", key.title())
            break

    rag_safety = tool_food_knowledge_retriever(f"{food_name} spoilage mold danger bacteria")
    
    return {
        "tool": "food_safety_protocol",
        "food": display_name,
        "condition": condition or "Unknown",
        "guideline": matched_guideline or "Inspect food physically: check for soft mushiness, discoloration, foul smell, or mold.",
        "scientific_protocol": rag_safety.get("content", ""),
        "sources": rag_safety.get("sources", [])
    }

# Tool 4: Nutrition Lookup Tool
def tool_nutrition_lookup(food_name: str) -> Dict[str, Any]:
    """Retrieves verified nutritional facts, calories, and macronutrient profile."""
    # First check optional external API
    external_api_key = os.environ.get("NUTRITION_API_KEY")
    external_api_url = os.environ.get("NUTRITION_API_URL")
    
    if external_api_key and external_api_url:
        try:
            resp = requests.get(
                external_api_url,
                params={"query": food_name},
                headers={"X-Api-Key": external_api_key},
                timeout=3.0
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "tool": "nutrition_lookup",
                    "source": "External Nutrition API",
                    "nutrition": data,
                    "found": True
                }
        except Exception:
            pass  # Fall through to verified local reference database
            
    metadata = get_classes_metadata().get("classes", {})
    food_clean = food_name.lower().strip()
    
    for key, data in metadata.items():
        aliases = data.get("detector_aliases", []) + [key, data.get("display_name", "").lower()]
        if any(alias in food_clean or food_clean in alias for alias in aliases):
            nutrition = data.get("nutrition")
            if nutrition:
                return {
                    "tool": "nutrition_lookup",
                    "source": "USDA FoodData Central Standards",
                    "food": data.get("display_name"),
                    "nutrition": nutrition,
                    "found": True
                }

    return {
        "tool": "nutrition_lookup",
        "source": "Unavailable",
        "food": food_name.title(),
        "found": False,
        "message": f"Specific nutritional breakdown for {food_name} is not currently indexed."
    }

# Tool 5: Current Scan Context Tool
def tool_current_scan_context(scan_context: Dict[str, Any]) -> Dict[str, Any]:
    """Inspects the client's current scanned image context."""
    if not scan_context:
        return {"tool": "current_scan_context", "active": False}
        
    return {
        "tool": "current_scan_context",
        "active": True,
        "item": scan_context.get("item"),
        "category": scan_context.get("category"),
        "freshness": scan_context.get("freshness"),
        "confidence": scan_context.get("confidence") or scan_context.get("detection_confidence"),
        "freshness_confidence": scan_context.get("freshness_confidence")
    }

# Tool 6: Scan History Tool
def tool_scan_history(limit: int = 5) -> Dict[str, Any]:
    """Retrieves user's recent scan history records."""
    records = history_service.get_all(limit=limit)
    return {
        "tool": "scan_history",
        "count": len(records),
        "recent_scans": [
            {
                "id": r.get("id"),
                "timestamp": r.get("timestamp"),
                "primary_item": r.get("primary_item"),
                "freshness": r.get("freshness")
            }
            for r in records
        ]
    }
