"""
FreshLens AI - Multilingual Prompt Templates & Safety Guardrails
Includes system prompts, safety disclaimers, and contextual formatting
supporting English, Hindi, and regional languages.
"""
from typing import Dict, Any, Optional

SAFETY_DISCLAIMER_EN = (
    "\n\n*⚠️ Safety Note: AI visual assessments are informational estimates and cannot detect internal pathogens, "
    "bacterial toxins, or chemical spoilage. Always verify food physically (texture, smell, temperature) before consumption.*"
)

SAFETY_DISCLAIMER_HI = (
    "\n\n*⚠️ सुरक्षा नोट: एआई विज़ुअल मूल्यांकन केवल सूचनात्मक है और यह आंतरिक बैक्टीरिया, फंगल टॉक्सिन या रासायनिक खराबी का पता नहीं लगा सकता। खाने से पहले हमेशा गंध, बनावट और स्वाद की जांच करें।* "
)

SYSTEM_PROMPT_EN = (
    "You are FreshLens AI Assistant, an expert in food safety, produce freshness, "
    "refrigeration science, and post-harvest storage techniques. "
    "Your objective is to provide actionable, grounded, and scientifically verified food safety guidance. "
    "Guidelines:\n"
    "1. Ground your answers in USDA, FDA, and verified postharvest standards.\n"
    "2. If the user refers to 'this food' or 'it', inspect the active scan context provided.\n"
    "3. Never make absolute medical diagnoses or claim food is 100% sterile or guaranteed safe.\n"
    "4. For spoiled produce, recommend safe disposal and cross-contamination prevention.\n"
    "5. Keep responses concise, well-structured, and helpful for consumers and kitchens."
)

SYSTEM_PROMPT_HI = (
    "आप FreshLens AI सहायक हैं—खाद्य सुरक्षा, फलों और सब्जियों के संरक्षण, और रेफ्रिजरेशन विज्ञान के विशेषज्ञ। "
    "आपका उद्देश्य वैज्ञानिक और विश्वसनीय खाद्य सुरक्षा सलाह देना है। "
    "दिशानिर्देश:\n"
    "1. हमेशा वैज्ञानिक और सत्यापित मानकों (USDA/FSSAI) के आधार पर उत्तर दें।\n"
    "2. यदि उपयोगकर्ता 'यह फल' या 'यह खाना' कहता है, तो स्कैन संदर्भ का उपयोग करें।\n"
    "3. खराब या सड़े हुए भोजन के लिए सुरक्षित निपटान (disposal) और रोकथाम की सलाह दें।\n"
    "4. जवाब स्पष्ट, सम्मानजनक और प्राकृतिक हिंदी में दें।"
)

def build_agent_prompt(
    user_query: str,
    language: str = "en",
    scan_context: Optional[Dict[str, Any]] = None,
    rag_context: str = "",
    tool_data: Optional[Dict[str, Any]] = None
) -> str:
    """Builds a structured prompt for the agent reasoning engine."""
    scan_block = ""
    if scan_context and scan_context.get("item"):
        item = scan_context.get("item")
        category = scan_context.get("category", "Food")
        freshness = scan_context.get("freshness", "Unknown")
        conf = scan_context.get("confidence") or scan_context.get("detection_confidence")
        conf_str = f"{round(float(conf)*100, 1)}%" if conf else "Calculated"
        scan_block = f"\n[Active Scan Context: Detected {item} ({category}), Freshness: {freshness}, Confidence: {conf_str}]\n"

    rag_block = f"\n[Retrieved Scientific Knowledge]:\n{rag_context}\n" if rag_context else ""
    tool_block = f"\n[Tool Insights]:\n{str(tool_data)}\n" if tool_data else ""

    return f"User Query: {user_query}\nLanguage: {language}\n{scan_block}{rag_block}{tool_block}"
