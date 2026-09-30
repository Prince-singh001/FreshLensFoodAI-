import os
import json
import uuid
import shutil
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    from config import FEEDBACK_FILE, FEEDBACK_UPLOAD_DIR, UPLOAD_FOLDER
    from utils.logger import logger
except ImportError:
    from backend.config import FEEDBACK_FILE, FEEDBACK_UPLOAD_DIR, UPLOAD_FOLDER
    from backend.utils.logger import logger

_feedback_lock = threading.Lock()

class FeedbackService:
    def __init__(self, feedback_file: str = FEEDBACK_FILE, upload_dir: str = FEEDBACK_UPLOAD_DIR):
        self.feedback_file = feedback_file
        self.upload_dir = upload_dir
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def submit_feedback(
        self,
        image_url: str,
        predicted_class: str,
        correct_class: str,
        user_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Securely queues feedback into a review dataset without mutating production model weights.
        """
        feedback_id = str(uuid.uuid4())[:12]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Resolve image from static uploads
        base_name = os.path.basename(image_url)
        source_path = os.path.join(UPLOAD_FOLDER, base_name)

        saved_path = ""
        if os.path.exists(source_path):
            target_filename = f"{feedback_id}_{base_name}"
            target_path = os.path.join(self.upload_dir, target_filename)
            try:
                shutil.copy2(source_path, target_path)
                saved_path = target_path
            except Exception as e:
                logger.error(f"Failed to copy image for feedback review: {e}")

        feedback_entry = {
            "id": feedback_id,
            "timestamp": timestamp,
            "image_url": image_url,
            "saved_image_path": saved_path,
            "predicted_class": predicted_class,
            "correct_class": correct_class,
            "user_notes": user_notes or "",
            "review_status": "pending",
            "curated_for_retraining": False
        }

        with _feedback_lock:
            try:
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception:
                entries = []

            entries.append(feedback_entry)

            temp_file = f"{self.feedback_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)
            os.replace(temp_file, self.feedback_file)

        logger.info(f"Feedback entry {feedback_id} recorded for future batch retraining: {predicted_class} -> {correct_class}")

        return {
            "success": True,
            "feedback_id": feedback_id,
            "status": "queued_for_review",
            "message": "Thank you! Your feedback has been securely submitted to our quality queue for review and future batch retraining."
        }

    def get_pending(self) -> List[Dict[str, Any]]:
        with _feedback_lock:
            if not os.path.exists(self.feedback_file):
                return []
            try:
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                return [e for e in entries if e.get("review_status") == "pending"]
            except Exception:
                return []

feedback_service = FeedbackService()
