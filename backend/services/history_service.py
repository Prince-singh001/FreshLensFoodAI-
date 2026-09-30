import os
import json
import uuid
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from config import HISTORY_FILE, DATA_DIR
    from utils.logger import logger
except ImportError:
    from backend.config import HISTORY_FILE, DATA_DIR
    from backend.utils.logger import logger

_history_lock = threading.Lock()

class HistoryService:
    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file = history_file
        self._ensure_storage_ready()

    def _ensure_storage_ready(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def get_all(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with _history_lock:
            if not os.path.exists(self.history_file):
                return []
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data[offset:offset + limit]
            except Exception as e:
                logger.error(f"Error loading scan history: {e}")
                return []

    def count(self) -> int:
        with _history_lock:
            if not os.path.exists(self.history_file):
                return 0
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return len(data)
            except Exception:
                return 0

    def add_record(self, scan_data: Dict[str, Any]) -> Dict[str, Any]:
        with _history_lock:
            try:
                if os.path.exists(self.history_file):
                    with open(self.history_file, "r", encoding="utf-8") as f:
                        records = json.load(f)
                else:
                    records = []
            except Exception:
                records = []

            record = {
                "id": scan_data.get("id") or str(uuid.uuid4())[:8],
                "timestamp": scan_data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "food_name": scan_data.get("food_name", "Unknown"),
                "condition": scan_data.get("condition", "Not Available"),
                "confidence": scan_data.get("confidence", 0.0),
                "category": scan_data.get("detected_category") or scan_data.get("category", "General"),
                "selected_category": scan_data.get("selected_category", "All"),
                "detected_category": scan_data.get("detected_category", "General"),
                "image_url": scan_data.get("image_url", ""),
                "annotated_image_url": scan_data.get("annotated_image_url", ""),
                "total_objects": scan_data.get("total_objects", 1),
                "objects": scan_data.get("objects", []),
                "warning": scan_data.get("warning", ""),
                "stability_warning": scan_data.get("stability_warning", "")
            }

            records.insert(0, record)
            # Retain maximum 100 recent scans for local storage
            records = records[:100]

            temp_file = f"{self.history_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            os.replace(temp_file, self.history_file)

            return record

    def delete_by_id(self, record_id: str) -> bool:
        with _history_lock:
            if not os.path.exists(self.history_file):
                return False
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    records = json.load(f)

                filtered = [r for r in records if r.get("id") != record_id]
                if len(filtered) == len(records):
                    return False

                temp_file = f"{self.history_file}.tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(filtered, f, indent=2)
                os.replace(temp_file, self.history_file)
                return True
            except Exception as e:
                logger.error(f"Error deleting record {record_id}: {e}")
                return False

    def clear(self) -> bool:
        with _history_lock:
            try:
                with open(self.history_file, "w", encoding="utf-8") as f:
                    json.dump([], f)
                return True
            except Exception as e:
                logger.error(f"Error clearing history: {e}")
                return False

history_service = HistoryService()
