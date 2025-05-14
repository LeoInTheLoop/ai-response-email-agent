import os
import json
from typing import List, Dict, Optional

class SummaryDataHelper:
    def __init__(self, filename: str):
        """
        Load summary data from a specific file, such as 'matador_tone.json'.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(base_dir, "../../data/summary_data"))
        filepath = os.path.join(data_dir, filename)

        self.data: List[Dict] = []
        self._load_file(filepath)

    def _load_file(self, filepath: str):
        """Load a single JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if isinstance(json_data, dict):
                    json_data = [json_data]
                self.data.extend(json_data)
        except Exception as e:
            print(f"[WARN] Failed to load {filepath}: {e}")

    def get_by_key(self, key: str, value: str) -> Optional[Dict]:
        """Get first match by exact value for a specific key (e.g., 'tone', 'email', 'project')."""
        for entry in self.data:
            if entry.get(key, "").lower() == value.lower():
                return entry
        return None

    def find_best_match(self, key: str, query: str) -> Optional[Dict]:
        """Match loosely based on keyword lists (if available)."""
        for entry in self.data:
            if any(keyword.lower() in query.lower() for keyword in entry.get("keywords", [])):
                if key in entry:
                    return entry
        return None

    def get_all(self) -> List[Dict]:
        return self.data
