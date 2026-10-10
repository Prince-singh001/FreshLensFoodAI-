"""
FoodLens-AI - RAG Knowledge Base & Reference Documentation
Verified food safety, storage recommendations, shelf-life benchmarks,
and spoilage protocols from USDA FSIS, FDA, and WHO standards.
"""
from typing import List, Dict, Any

FOOD_KNOWLEDGE_DOCS: List[Dict[str, Any]] = [
    {
        "id": "usda_danger_zone",
        "title": "The Temperature Danger Zone for Food Safety",
        "source": "USDA Food Safety and Inspection Service (FSIS)",
        "url": "https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/danger-zone-40f-140f",
        "category": "food_safety",
        "last_updated": "2024",
        "content": (
            "Bacteria grow most rapidly in the range of temperatures between 40°F (4.4°C) and 140°F (60°C), "
            "doubling in number in as little as 20 minutes. This range is called the 'Temperature Danger Zone'. "
            "Perishable cooked foods, cut produce, and prepared foods should never be left at room temperature "
            "for more than 2 hours (or 1 hour if the ambient temperature is above 90°F / 32°C). "
            "Always refrigerate perishable foods promptly at or below 40°F (4°C) to inhibit bacterial replication."
        )
    },
    {
        "id": "fda_produce_handling",
        "title": "Safe Handling of Raw Fruits and Vegetables",
        "source": "U.S. Food and Drug Administration (FDA)",
        "url": "https://www.fda.gov/food/buy-store-serve-safe-food/selecting-and-serving-fresh-fruits-and-vegetables-safely",
        "category": "produce_handling",
        "last_updated": "2024",
        "content": (
            "Always wash hands with warm water and soap for at least 20 seconds before and after handling fresh produce. "
            "Rinse fresh fruits and vegetables thoroughly under running tap water before eating, peeling, or cooking. "
            "Never use soap, detergent, or bleach on produce because the peel is porous and can absorb chemicals. "
            "Firm produce such as potatoes, carrots, and cucumbers should be scrubbed with a clean produce brush. "
            "Always cut away damaged or bruised areas on fresh fruits and vegetables before preparing or eating."
        )
    },
    {
        "id": "usda_mold_guidelines",
        "title": "Molds on Food: Are They Dangerous?",
        "source": "USDA Food Safety and Inspection Service (FSIS)",
        "url": "https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/molds-food-are-they-dangerous",
        "category": "mold_spoilage",
        "last_updated": "2024",
        "content": (
            "Molds produce microscopic roots (hyphae) that penetrate deep into soft foods beyond what is visible on the surface. "
            "1. Soft fruits and vegetables (e.g. tomatoes, peaches, strawberries, cucumbers, bananas): "
            "DISCARD COMPLETELY if surface mold is visible, as deep fungal hyphae and mycotoxins spread rapidly through soft flesh. "
            "2. Firm fruits and vegetables (e.g. carrots, bell peppers, cabbage, apples, potatoes): "
            "You may cut off at least 1 inch (2.5 cm) around and below the mold spot, keeping the knife out of the mold itself. "
            "Never sniff moldy items, as inhaling fungal spores can trigger severe respiratory reactions."
        )
    },
    {
        "id": "produce_ethylene_separation",
        "title": "Ethylene Gas Sensitivity and Produce Storage Separation",
        "source": "Postharvest Technology Center / UC Davis & USDA",
        "url": "https://postharvest.ucdavis.edu/",
        "category": "storage_techniques",
        "last_updated": "2023",
        "content": (
            "Certain fruits produce high quantities of ethylene gas, a natural plant hormone that accelerates ripening and decay. "
            "High ethylene producers include: Apples, Bananas, Melons, Tomatoes, and Avocados. "
            "Ethylene-sensitive produce includes: Cucumbers, Potatoes, Carrots, Leafy Greens, Broccoli, and Onions. "
            "Never store ethylene-producers (such as apples or bananas) in the same bin or sealed bag with ethylene-sensitive items "
            "(such as cucumbers or potatoes). For instance, storing onions and potatoes together causes potatoes to sprout and onions to soften rapidly."
        )
    },
    {
        "id": "apple_storage_profile",
        "title": "Apple Freshness, Cold Storage & Shelf-Life Protocols",
        "source": "USDA Agricultural Research Service (ARS)",
        "url": "https://www.ars.usda.gov/",
        "category": "item_storage",
        "last_updated": "2024",
        "content": (
            "Apples maintain peak firmness and nutritional density when stored in the refrigerator crisper drawer at 32°F–38°F (0°C–3°C) "
            "with high humidity (85-90%). Apples stored in the refrigerator stay crisp for 4 to 8 weeks, whereas at room temperature they soften "
            "up to 10 times faster within 1 to 2 weeks. Visual indicators of spoilage include: deep brown soft spots, sunken skin rot, wrinkled leather-like texture, "
            "or a fermented alcoholic odor. Bruised apples release accelerated ethylene gas, so separate bruised units from sound fruit."
        )
    },
    {
        "id": "banana_storage_profile",
        "title": "Banana Ripening Phases and Spoilage Prevention",
        "source": "International Tropical Fruits Network (TFNet) & FAO",
        "url": "https://www.fao.org/",
        "category": "item_storage",
        "last_updated": "2024",
        "content": (
            "Bananas should be kept at room temperature (65°F–68°F / 18°C–20°C) with stem ends intact until desired ripeness is reached. "
            "Wrapping banana crown stems in plastic film reduces ethylene diffusion and extends shelf-life by 3–5 days. "
            "Refrigeration turns banana peels dark brown due to chill injury, but the internal fruit remains firm and edible for several additional days. "
            "Signs of spoilage: Black mushy internal pulp, leaking liquid from the peel, fermented sour odor, or white mold around the stem crown. "
            "Overripe bananas with dark specks are safe for baking, but bananas with mold or foul odor must be discarded."
        )
    },
    {
        "id": "tomato_storage_profile",
        "title": "Tomato Flavor Preservation and Spoilage Identification",
        "source": "University of Florida Postharvest Institute & USDA",
        "url": "https://edis.ifas.ufl.edu/",
        "category": "item_storage",
        "last_updated": "2024",
        "content": (
            "Unripe or ripe fresh tomatoes should be stored stem-side down at room temperature (60°F–70°F / 15°C–21°C) away from direct sunlight. "
            "Chilling fresh tomatoes below 55°F (12.8°C) permanently deactivates flavor enzyme synthesis and causes a mealy, mushy texture. "
            "Only refrigerate fully ripe tomatoes if they cannot be consumed immediately, and allow them to return to room temperature before serving. "
            "Signs of spoilage: Water-soaked soft depressions, black or white surface mold, split weeping skin, or unpleasant sour smell. "
            "Because tomatoes are soft and acidic, any mold growth penetrates the entire fruit; moldy tomatoes must be thrown away completely."
        )
    },
    {
        "id": "potato_storage_profile",
        "title": "Potato Solanine Toxicity and Storage Guidelines",
        "source": "FDA & European Food Safety Authority (EFSA)",
        "url": "https://www.fda.gov/",
        "category": "item_storage",
        "last_updated": "2024",
        "content": (
            "Store potatoes in a cool, dark, well-ventilated location (45°F–50°F / 7°C–10°C). Never refrigerate raw potatoes because cold temperatures "
            "convert potato starch into reducing sugars (cold-induced sweetening), producing higher levels of acrylamide during frying or baking. "
            "Exposure to light stimulates chlorophyll and toxic glycoalkaloids (solanine and chaconine). "
            "Green skin or bitter sprouts indicate solanine accumulation, which causes gastrointestinal cramps and nausea. "
            "Cut away minor green patches and sprouts; discard potatoes that are heavily green, spongy, soft, or smell musty."
        )
    },
    {
        "id": "cucumber_storage_profile",
        "title": "Cucumber Chilling Injury and Humidity Control",
        "source": "USDA Agricultural Marketing Service (AMS)",
        "url": "https://www.ams.usda.gov/",
        "category": "item_storage",
        "last_updated": "2023",
        "content": (
            "Cucumbers are sensitive to chilling injury if held below 50°F (10°C) for more than 2 to 3 days. "
            "Store cucumbers in the warmest part of the refrigerator (such as the upper shelf or crisper set to moderate humidity) for up to 1 week. "
            "Keep cucumbers dry; excess surface moisture promotes bacterial soft rot (Pseudomonas). "
            "Keep cucumbers separated from high ethylene emitters (apples, tomatoes, bananas) which trigger yellowing and accelerated decay. "
            "Signs of spoilage include yellowing skin, pitted watery depressions, soft mushy tips, and milky or slippery slime on the rind."
        )
    },
    {
        "id": "orange_citrus_storage",
        "title": "Citrus Fruit Longevity and Mold Management",
        "source": "Citrus Research Board & USDA",
        "url": "https://www.usda.gov/",
        "category": "item_storage",
        "last_updated": "2024",
        "content": (
            "Oranges and citrus maintain good quality at room temperature for 1 week and in the refrigerator crisper drawer (40°F / 4°C) for 3 to 4 weeks. "
            "Store citrus in mesh bags or breathable containers; sealed plastic bags trap moisture and foster green or blue mold (Penicillium digitatum). "
            "Signs of citrus spoilage: Powdery blue-green mold crusts, fermented alcohol smell, or soft spongy sunken rind spots. "
            "If an orange develops green or blue mold, discard it immediately and sanitize adjacent fruits in the container."
        )
    },
    {
        "id": "prepared_food_safety",
        "title": "Cooked Leftovers and Prepared Food Safety Timeline",
        "source": "USDA FSIS Food Safety Education",
        "url": "https://www.fsis.usda.gov/",
        "category": "prepared_food",
        "last_updated": "2024",
        "content": (
            "Cooked leftovers (such as rice, pasta, curries, cooked vegetables, and meats) must be sealed in airtight containers and refrigerated "
            "within 2 hours of cooking. Cooked foods should be consumed within 3 to 4 days when stored at <= 40°F (4°C). "
            "Cooked rice left at room temperature is vulnerable to Bacillus cereus spores, which produce heat-resistant emetic toxins that survive reheating. "
            "Visual inspection cannot detect bacterial toxins in prepared foods; adhere strictly to the 3-to-4-day rule. When in doubt, throw it out."
        )
    }
]

def get_knowledge_corpus() -> List[Dict[str, Any]]:
    """Returns the complete list of curated food safety and storage documents."""
    return FOOD_KNOWLEDGE_DOCS
