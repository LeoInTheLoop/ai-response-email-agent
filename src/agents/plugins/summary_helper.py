import os
import json
from typing import List, Dict, Optional
from config import SUMMARY_DATA_PATH


class SummaryDataHelper:
    def __init__(self, data_path: str = SUMMARY_DATA_PATH):
        """
        Load summary data from a specified path.
        
        Args:
            data_path: Path to the directory or file containing summary data
        """
        self.data: List[Dict] = []
        

        # Check if it's a directory or file

        if SUMMARY_DATA_PATH.is_file():
            self._load_file(SUMMARY_DATA_PATH)
        else:
            # Treat as a single file
            print(f"[INFO] can't Loading summary data from directory: {data_path}")
    
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
    
    def find_best_match(self, query: str) -> Optional[Dict]:
        """
        Find the best matching entry based on keywords in the query.
        
        Args:
            query: The text to match against keywords
            
        Returns:
            The best matching entry or None if no match is found
        """
        for entry in self.data:
            if "keywords" in entry and any(keyword.lower() in query.lower() for keyword in entry.get("keywords", [])):
                return entry
        return None
    
    def get_all(self) -> List[Dict]:
        return self.data