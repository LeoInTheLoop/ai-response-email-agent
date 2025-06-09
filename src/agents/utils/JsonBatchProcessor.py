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
    

    def df_to_text(df: pd.DataFrame) -> str:
        lines = []
        for idx, row in df.iterrows():
            subj = str(row.get("subject", "")).strip()
            body = str(row.get("body", "")).strip()
            lines.append(f"--- Email {len(lines) + 1} ---")
            lines.append(f"Subject: {subj}")
            lines.append("Body:")
            lines.append(body)
            lines.append("")
        return "\n".join(lines)
    
    async def batch_with_halving(
        agent: object,  # The agent instance passed in
        batch_df: pd.DataFrame,
        known_projects: list[dict],
        current_batch_size: int,
        min_batch_size: int = 1,
        recursion_depth: int = 0,
        max_recursion: int = 10
    ) -> str:
        """
        Process batch recursively, halving batch size when encountering size-related errors.
        
        Args:
            agent: Pre-configured agent instance
            batch_df: DataFrame containing email batch to process
            known_projects: List of known projects for reference
            current_batch_size: Current batch size being attempted
            min_batch_size: Minimum batch size to attempt (default 1)
            recursion_depth: Current recursion depth (default 0)
            max_recursion: Maximum allowed recursion depth (default 10)
            
        Returns:
            Combined JSON response from successful processing
        """
        if recursion_depth >= max_recursion:
            print(f"⚠️ Max recursion depth ({max_recursion}) reached with batch size {current_batch_size}")
            return "[]"
        
        if current_batch_size < min_batch_size:
            print(f"⚠️ Reached minimum batch size {min_batch_size}")
            return "[]"
        
        # Prepare the input data
        current_batch = batch_df.head(current_batch_size)
        emails_text = df_to_text(current_batch)
        prompt_template = format_batch(current_batch, known_projects)
        
        # Update agent instructions with current batch
        agent.update_instructions(prompt_template)
        
        history = ChatHistory()
        history.add_user_message(emails_text)
        
        try:
            # Attempt processing
            raw_response = ""
            async for part in agent.invoke_stream(history):
                if getattr(part, "content", "").strip():
                    raw_response += part.content
            
            if not raw_response:
                print(f"Empty response with batch size {current_batch_size}")
                return "[]"
                
            return raw_response
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error with batch size {current_batch_size}: {error_msg}")
            
            # Check if error is size-related
            if "Request body too large" in error_msg or "too long" in error_msg.lower():
                new_batch_size = max(min_batch_size, current_batch_size // 2)
                print(f"Halving batch size from {current_batch_size} to {new_batch_size}")
                
                # Process first half
                first_half = await batch_with_halving(
                    agent,
                    batch_df.iloc[:current_batch_size//2],
                    known_projects,
                    new_batch_size,
                    min_batch_size,
                    recursion_depth + 1,
                    max_recursion
                )
                
                # Process second half
                second_half = await batch_with_halving(
                    agent,
                    batch_df.iloc[current_batch_size//2:current_batch_size],
                    known_projects,
                    new_batch_size,
                    min_batch_size,
                    recursion_depth + 1,
                    max_recursion
                )
                
                # Combine results
                try:
                    first_json = json.loads(first_half) if first_half else []
                    second_json = json.loads(second_half) if second_half else []
                    combined = first_json + second_json
                    return json.dumps(combined)
                except json.JSONDecodeError:
                    print("Failed to combine JSON responses")
                    return "[]"
            
            # For non-size-related errors, return empty array
            print("Non-size-related error encountered")
            return "[]"
        

