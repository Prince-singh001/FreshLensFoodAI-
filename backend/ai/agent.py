"""
FoodLens-AI - Agentic AI Orchestrator
Uses LangChain patterns, tool selection, RAG retrieval, scan-context integration,
and multilingual synthesis (English, Hindi, etc.) with conversational memory.
"""
import re
import uuid
from typing import Dict, Any, List, Optional

from .retriever import get_retriever
from .memory import get_memory
from .llm import get_llm_client
from .prompts import (
    SAFETY_DISCLAIMER_EN,
    SAFETY_DISCLAIMER_HI,
    SYSTEM_PROMPT_EN,
    SYSTEM_PROMPT_HI
)
from .tools import (
    tool_food_knowledge_retriever,
    tool_storage_recommendation,
    tool_food_safety_protocol,
    tool_nutrition_lookup,
    tool_current_scan_context,
    tool_scan_history
)

class FoodLensAgent:
    """Agentic AI coordinator for food recognition, freshness, storage and safety."""
    def __init__(self):
        self.memory = get_memory()
        self.llm = get_llm_client()
        self.retriever = get_retriever()

    def process_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        language: str = "en",
        scan_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        cid = conversation_id or str(uuid.uuid4())
        msg_clean = message.strip()
        msg_lower = msg_clean.lower()
        lang = language.lower() if language else "en"

        # Auto-detect Devanagari Hindi script if language is not explicitly set
        if any('\u0900' <= char <= '\u097f' for char in msg_clean):
            lang = "hi"

        # Record user message in memory
        self.memory.add_message(cid, "user", msg_clean)

        # 1. Identify Contextual Item (Scan Context or Query Mentions)
        active_item = None
        if scan_context and scan_context.get("item"):
            active_item = scan_context.get("item")

        # Check if the query itself mentions a known produce item
        known_produce = ["apple", "banana", "orange", "tomato", "potato", "cucumber", "bitter gourd", "karela", "onion", "carrot", "broccoli", "rice", "bread", "pizza", "sandwich", "mango", "grapes"]
        for p in known_produce:
            if p in msg_lower:
                active_item = p.title()
                break

        # Check Hindi terms for produce
        hindi_produce_map = {
            "सेब": "Apple",
            "केला": "Banana",
            "संतरा": "Orange",
            "टमाटर": "Tomato",
            "आलू": "Potato",
            "खीरा": "Cucumber",
            "करेला": "Bitter Gourd",
            "प्याज": "Onion",
            "गाजर": "Carrot",
            "चावल": "Rice",
            "रोटी": "Bread"
        }
        for hi_word, en_name in hindi_produce_map.items():
            if hi_word in msg_clean:
                active_item = en_name
                break

        # 2. Tool Selection based on Intent
        selected_tool_name = "general_knowledge"
        tool_output: Dict[str, Any] = {}
        sources: List[Dict[str, str]] = []

        # Intent A: Nutrition or Calorie query
        if any(w in msg_lower for w in ["nutrition", "calories", "calorie", "protein", "carbs", "fiber", "vitamins", "पोषण", "कैलोरी"]):
            selected_tool_name = "nutrition_lookup"
            target_food = active_item or "Apple"
            tool_output = tool_nutrition_lookup(target_food)

        # Intent B: Current Scan context query ("What did you detect?", "What is in the image?", "Scan result")
        elif any(w in msg_lower for w in ["what did you detect", "what is this", "scan result", "what food", "क्या मिला", "स्कैन"]):
            selected_tool_name = "current_scan_context"
            tool_output = tool_current_scan_context(scan_context)

        # Intent C: Storage advice ("How to store", "How should I store", "Keep fresh", "Fridge", "स्टोर", "रखना")
        elif any(w in msg_lower for w in ["store", "storage", "keep", "fridge", "refrigerator", "shelf life", "स्टोर", "रखना", "रखें"]):
            selected_tool_name = "storage_recommendation"
            target_food = active_item or "Produce"
            tool_output = tool_storage_recommendation(target_food)
            if tool_output.get("sources"):
                sources.extend(tool_output["sources"])

        # Intent D: Safety, Spoilage, Mold, or "Can I eat it?"
        elif any(w in msg_lower for w in ["safe", "eat", "spoil", "spoiled", "mold", "fungus", "bad", "sick", "toxic", "सुरक्षित", "खा सकते", "सड़ा", "खराब"]):
            selected_tool_name = "food_safety_protocol"
            target_food = active_item or (scan_context.get("item") if scan_context else "Produce")
            cond = scan_context.get("freshness") if scan_context else None
            tool_output = tool_food_safety_protocol(target_food, cond)
            if tool_output.get("sources"):
                sources.extend(tool_output["sources"])

        # Intent E: General RAG retrieval
        else:
            selected_tool_name = "food_knowledge_retriever"
            rag_res = tool_food_knowledge_retriever(msg_clean)
            tool_output = rag_res
            if rag_res.get("sources"):
                sources.extend(rag_res["sources"])

        # Always fetch top relevant scientific context if sources are empty
        if not sources:
            default_rag = tool_food_knowledge_retriever(f"{active_item or 'food'} freshness safety storage")
            if default_rag.get("sources"):
                sources = default_rag["sources"]

        # 3. Response Generation (Grounded, Multilingual)
        answer = self._generate_response(
            query=msg_clean,
            tool_name=selected_tool_name,
            tool_output=tool_output,
            active_item=active_item,
            scan_context=scan_context,
            language=lang,
            sources=sources
        )

        # Record assistant answer in memory
        self.memory.add_message(cid, "assistant", answer)

        return {
            "success": True,
            "conversation_id": cid,
            "language": lang,
            "answer": answer,
            "tool_used": selected_tool_name,
            "sources": sources
        }

    def _generate_response(
        self,
        query: str,
        tool_name: str,
        tool_output: Dict[str, Any],
        active_item: Optional[str],
        scan_context: Optional[Dict[str, Any]],
        language: str,
        sources: List[Dict[str, str]]
    ) -> str:
        """Synthesizes the final grounded answer in the requested language."""
        is_hindi = language == "hi"
        disclaimer = SAFETY_DISCLAIMER_HI if is_hindi else SAFETY_DISCLAIMER_EN

        # A: Nutrition Response
        if tool_name == "nutrition_lookup":
            nutrition = tool_output.get("nutrition")
            food = tool_output.get("food", active_item or "Produce")
            if nutrition and isinstance(nutrition, dict) and "calories_per_100g" in nutrition:
                if is_hindi:
                    return (
                        f"**{food} के पोषण संबंधी तथ्य (प्रति 100 ग्राम):**\n\n"
                        f"- **कैलोरी (Calories):** {nutrition.get('calories_per_100g')} kcal\n"
                        f"- **कार्बोहाइड्रेट:** {nutrition.get('carbs_g')}g\n"
                        f"- **प्रोटीन:** {nutrition.get('protein_g')}g\n"
                        f"- **फाइबर:** {nutrition.get('fiber_g')}g\n"
                        f"- **प्रमुख विटामिन एवं खनिज:** {nutrition.get('key_vitamins')}\n\n"
                        f"*स्रोत: {tool_output.get('source')}*"
                        + disclaimer
                    )
                else:
                    return (
                        f"**Nutritional Breakdown for {food} (per 100g):**\n\n"
                        f"- **Calories:** {nutrition.get('calories_per_100g')} kcal\n"
                        f"- **Carbohydrates:** {nutrition.get('carbs_g')}g\n"
                        f"- **Protein:** {nutrition.get('protein_g')}g\n"
                        f"- **Dietary Fiber:** {nutrition.get('fiber_g')}g\n"
                        f"- **Key Micronutrients:** {nutrition.get('key_vitamins')}\n\n"
                        f"*Reference Source: {tool_output.get('source')}*"
                        + disclaimer
                    )
            else:
                msg = f"Specific nutritional figures for {food} are not currently indexed." if not is_hindi else f"{food} के सटीक पोषण आंकड़े वर्तमान में डेटाबेस में उपलब्ध नहीं हैं।"
                return msg + disclaimer

        # B: Scan Context Inquiry
        if tool_name == "current_scan_context":
            if tool_output.get("active"):
                item = tool_output.get("item", "Food Item")
                cat = tool_output.get("category", "General")
                fresh = tool_output.get("freshness", "Analyzed")
                conf = tool_output.get("confidence")
                conf_pct = f"{round(float(conf)*100, 1)}%" if conf else "high certainty"

                if is_hindi:
                    return (
                        f"**हालिया स्कैन परिणाम:**\n\n"
                        f"- **पहचाना गया खाद्य:** {item}\n"
                        f"- **श्रेणी:** {cat}\n"
                        f"- **ताज़गी की स्थिति:** {fresh}\n"
                        f"- **मॉडल विश्वसनीयता (Confidence):** {conf_pct}\n\n"
                        f"आप मुझसे इस {item} के भंडारण के तरीके या सुरक्षा से संबंधित कोई भी सवाल पूछ सकते हैं।"
                        + disclaimer
                    )
                else:
                    return (
                        f"**Active Scan Insights:**\n\n"
                        f"- **Identified Food:** {item}\n"
                        f"- **Category:** {cat}\n"
                        f"- **Freshness Evaluation:** {fresh}\n"
                        f"- **Model Confidence:** {conf_pct}\n\n"
                        f"Feel free to ask follow-up questions regarding storage recommendations or shelf-life for this {item}."
                        + disclaimer
                    )
            else:
                msg = "No active scan was detected. Please scan or upload a food image first to discuss its specific freshness." if not is_hindi else "कोई सक्रिय स्कैन नहीं मिला। ताज़गी की विस्तृत जानकारी के लिए कृपया पहले किसी फल या खाद्य की तस्वीर स्कैन करें।"
                return msg

        # C: Storage Recommendation
        if tool_name == "storage_recommendation":
            food = tool_output.get("food", active_item or "Produce")
            tip = tool_output.get("storage_tip", "Keep in a cool, clean environment.")
            if is_hindi:
                return (
                    f"**{food} के लिए भंडारण एवं संरक्षण सलाह:**\n\n"
                    f"{tip}\n\n"
                    f"**महत्वपूर्ण बातें:**\n"
                    f"- फलों को धोने के बाद पूरी तरह सुखाकर ही फ्रिज में रखें।\n"
                    f"- अत्यधिक एथिलीन छोड़ने वाले फलों (जैसे सेब, केला) को खीरे और आलू से दूर रखें।"
                    + disclaimer
                )
            else:
                return (
                    f"**Storage & Shelf-Life Protocol for {food}:**\n\n"
                    f"{tip}\n\n"
                    f"**Key Post-Harvest Best Practices:**\n"
                    f"- Ensure produce is completely dry before refrigerated storage to prevent fungal rot.\n"
                    f"- Isolate ethylene-producing fruits (such as apples and bananas) from ethylene-sensitive vegetables."
                    + disclaimer
                )

        # D: Food Safety Protocol
        if tool_name == "food_safety_protocol":
            food = tool_output.get("food", active_item or "Produce")
            condition = tool_output.get("condition")
            guideline = tool_output.get("guideline")

            is_spoiled = condition and "spoil" in condition.lower()

            if is_hindi:
                if is_spoiled:
                    return (
                        f"**चेतावनी: {food} के उपभोग से बचें:**\n\n"
                        f"स्कैनर विश्लेषण के अनुसार यह {food} खराब (Spoiled) प्रतीत होता है।\n"
                        f"- **सुरक्षा निर्देश:** {guideline}\n"
                        f"- **फफूंद या सड़न का जोखिम:** फंगल मायकोटॉक्सिन और हानिकारक बैक्टीरिया खाद्य पदार्थों में गहरे फैल सकते हैं।\n"
                        f"- **सिफारिश:** इसे खाने से बचें और अन्य ताज़ा भोजन को संदूषण से बचाने के लिए इसे अलग करके फेंक दें।"
                        + disclaimer
                    )
                else:
                    return (
                        f"**{food} सुरक्षा एवं गुणवत्ता दिशानिर्देश:**\n\n"
                        f"{guideline}\n\n"
                        f"- हमेशा खाने से पहले सतह पर फफूंद, धब्बे, अत्यधिक नरमी या असामान्य खट्टी गंध की जांच करें।\n"
                        f"- संदेह होने पर सुरक्षित विकल्प चुनें और खराब हिस्से का सेवन न करें।"
                        + disclaimer
                    )
            else:
                if is_spoiled:
                    return (
                        f"**Advisory: Consumption Not Recommended for {food}:**\n\n"
                        f"Model analysis flagged visual degradation indicating this {food} is likely spoiled.\n"
                        f"- **Safety Protocol:** {guideline}\n"
                        f"- **Mycotoxin & Bacterial Risk:** Fungal hyphae penetrate deeply beyond surface mold in soft produce.\n"
                        f"- **Recommendation:** Safely discard to prevent cross-contamination with surrounding fresh foods."
                        + disclaimer
                    )
                else:
                    return (
                        f"**Food Safety & Spoilage Guidelines for {food}:**\n\n"
                        f"{guideline}\n\n"
                        f"- **Sensory Check:** Inspect for weeping skin, localized soft rot, or sour fermented aroma.\n"
                        f"- **Golden Rule:** When in doubt regarding perishables, throw it out."
                        + disclaimer
                    )

        # E: General Knowledge / RAG Fallback
        content = tool_output.get("content") or "FoodLens-AI is equipped to assist you with produce freshness, storage techniques, and food safety protocols."
        if is_hindi:
            return (
                "**FoodLens-AI खाद्य सुरक्षा अंतर्दृष्टि:**\n\n"
                f"{content}\n\n"
                "आप मुझसे किसी भी फल, सब्जी या भोजन के भंडारण, पोषक तत्वों या खराब होने के लक्षणों के बारे में पूछ सकते हैं।"
                + disclaimer
            )
        else:
            return (
                "**FoodLens-AI Scientific Guidance:**\n\n"
                f"{content}\n\n"
                "You can ask follow-up questions about specific storage temperatures, ethylene management, or nutrition facts."
                + disclaimer
            )

FreshLensAgent = FoodLensAgent

# Global singleton agent
_agent_instance = None

def get_agent() -> FoodLensAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FoodLensAgent()
    return _agent_instance
