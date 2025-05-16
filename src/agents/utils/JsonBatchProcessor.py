import os
import json
import re
import asyncio
import pandas as pd
from typing import List, Dict, Callable, Optional, Generator
from semantic_kernel import Kernel
import tiktoken
from semantic_kernel.contents import ChatHistory
from typing import Optional

from agents.utils.create_kernel_and_agent import create_agent

## do not delete This common 
# #class is designed to process batches of JSON data using an AI model. 
## It handles token counting, batch generation, and JSON extraction from AI responses.
## do not change unless debug 
## can add new features

class JsonBatchProcessor:
    def __init__(
        self,
        ai_model: str = "gpt-4o-mini",
        max_tokens: int = 8000,
        buffer_tokens: int = 4000,
        output_dir: str = "./output",
        output_prefix: str = "batch_",
        final_output_name: str = "merged_result.json"
    ):
        self.ai_model = ai_model
        self.max_tokens = max_tokens
        self.buffer_tokens = buffer_tokens
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.final_output_name = final_output_name

        self.enc = tiktoken.encoding_for_model(ai_model)
        os.makedirs(self.output_dir, exist_ok=True)

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def dynamic_batch_generator(
        self,
        df: pd.DataFrame,
        columns_for_tokenization: List[str],
        row_formatter: Callable[[Dict], str]
    ) -> Generator[pd.DataFrame, None, None]:
        batch, batch_tokens = [], 0

        for _, row in df.iterrows():
            row_text = row_formatter(row.to_dict())
            row_tokens = self.count_tokens(row_text)

            if batch_tokens + row_tokens > self.max_tokens - self.buffer_tokens:
                if batch:
                    yield pd.DataFrame(batch)
                batch, batch_tokens = [], 0

            batch.append(row.to_dict())
            batch_tokens += row_tokens

        if batch:
            yield pd.DataFrame(batch)

    def extract_json_from_response(self, response: str) -> List[Dict]:
        match = re.search(r"\[\s*{[\s\S]*?}\s*\]", response)
        if not match:
            print("No JSON array found.")
            return []
        try:
            data = json.loads(match.group(0).strip())
            return data if isinstance(data, list) else [data]
        except Exception as e:
            print(f"JSON parse error: {e}")
            return []

    def save_raw_output(self, response: str, batch_idx: int) -> str:
        path = os.path.join(self.output_dir, f"{self.output_prefix}{batch_idx}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(response)
        return path

    def merge_batch_results(self, batches: List[List[Dict]]) -> List[Dict]:
        seen, merged = set(), []
        for batch in batches:
            for item in batch:
                sig = json.dumps(item, sort_keys=True)
                if sig not in seen:
                    seen.add(sig)
                    merged.append(item)
        return merged

    def save_merged_results(self, results: List[Dict]) -> str:
        path = os.path.join(self.output_dir, self.final_output_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return path

