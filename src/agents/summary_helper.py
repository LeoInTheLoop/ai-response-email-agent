# summary_helper.py
import os
import json
from typing import List, Dict, Optional


class SummaryDataHelper:
    def __init__(self, data_dir: str):
        self.data: List[Dict] = []
        self._load_all_files(data_dir)

    def _load_all_files(self, data_dir: str):
        """Loads all JSON files in the directory"""
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        json_data = json.load(f)
                        # normalize: wrap dict as list for uniform access
                        if isinstance(json_data, dict):
                            json_data = [json_data]
                        self.data.extend(json_data)
                    except Exception as e:
                        print(f"[WARN] Failed to load {filename}: {e}")

    def get_summary_by_signature(self, signature: str) -> Optional[Dict]:
        """Find summary by exact signature (e.g., name in signature line)"""
        for entry in self.data:
            if entry.get("signature") == signature:
                return entry
        return None

    def get_summary_by_email(self, email: str) -> Optional[Dict]:
        """Stub: match email to a summary (can extend with a map or model)"""
        # If your summary file doesn’t contain emails, use a mapping table or embedding search
        return None

    def get_all_summaries(self) -> List[Dict]:
        return self.data

    def find_best_match(self, query: str) -> Optional[Dict]:
        """Find the best-matching summary entry using simple keyword match"""
        for entry in self.data:
            if any(keyword.lower() in query.lower() for keyword in entry.get("keywords", [])):
                return entry
        return None
