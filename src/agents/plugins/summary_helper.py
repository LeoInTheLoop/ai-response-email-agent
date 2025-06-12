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
            print(f"[SummaryDataHelper INFO] can't Loading summary data from directory: {data_path}")

        print(f"[SummaryDataHelper INFO] Loaded {len(self.data)} summary entries from {data_path}, first 10 entries: {self.data[:10]} /n /n ")
    
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
    
    
    def find_best_match(self, keylist: list[str]) -> Optional[Dict]:
        """
        Find the best matching entry based on keywords in the query.
        
        Args:
            query: The text to match against keywords
            
        Returns:
            The best matching entry or None if no match is found
        """
        role, tone, intent = keylist
        if not self.data:
            print("[WARN] No summary data loaded.")
            return None
        summary = []
        for entry in self.data:
            if (role in entry.get("tone", {}) or
                tone in entry["tone"].get(role, []) or
                intent in entry.get("intent")):
                summary.append(entry)
        
        return summary
    
    def get_all(self) -> List[Dict]:
        return self.data
    

# data example = [
#       {
#     "context": "Used in technical discussions, often involving detailed information or requests.",
#     "tone": {
#       "Client": [
#         "Formal",
#         "Technical"
#       ]
#     },
#     "intent": "Clarify / Request Info",
#     "example": "Explaining investment structures",
#     "greeting": [
#       "Mr. Buckner,"
#     ],
#     "closing": [
#       "Let me assure you, this is a real deal!!",
#       "Should you have additional questions, give me a call."
#     ],
#     "patterns": [
#       "I need your best...",
#       "For delivered gas behind..."
#     ],
#     "keywords": [
#       "gas price",
#       "microturbine",
#       "proposal"
#     ],
#     "signature": "Phillip Allen"
#   },
#   {
#     "context": "Formal communication regarding meetings and project updates.",
#     "tone": {
#       "Manager": [
#         "Formal",
#         "Direct"
#       ]
#     },
#     "intent": "Request Approval",
#     "example": "Asking for budget sign-off",
#     "greeting": [
#       "Please plan to attend",
#       "If you have any questions/conflicts, please feel free to call me."
#     ],
#     "closing": [
#       "Thanks,",
#       "Sincerely,"
#     ],
#     "patterns": [
#       "Please plan to attend the below Meeting:",
#       "If you have any questions/conflicts, please feel free to call me."
#     ],
#     "keywords": [
#       "Meeting",
#       "Please",
#       "Thank you"
#     ],
#     "signature": "Phillip"
#   },]