"""
JSON Data Extraction Module

This module provides reusable functions for extracting structured JSON data from
model responses and merging results from multiple training batches.
"""

import os
import json
import re
import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional, Generator, Tuple, Union, Callable
import tiktoken

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings


class DataExtractor:
    """
    A class to extract and merge structured JSON data from multiple model invocations.
    Handles batching of input data, invocation of agents, and aggregation of results.
    """

    def __init__(
        self,
        ai_model: str = "gpt-4o-mini",
        max_tokens: int = 7000,
        buffer_tokens: int = 1000,
        output_dir: str = "./output",
        output_prefix: str = "batch_",
        final_output_name: str = "merged_result.json"
    ):
        """
        Initialize the DataExtractor with configuration parameters.

        Args:
            ai_model (str, optional): Model ID to use for token counting. Defaults to "gpt-4o-mini".
            max_tokens (int, optional): Maximum tokens for a batch. Defaults to 7000.
            buffer_tokens (int, optional): Token buffer to avoid exceeding limits. Defaults to 1000.
            output_dir (str, optional): Directory to save outputs. Defaults to "./output".
            output_prefix (str, optional): Prefix for intermediate files. Defaults to "batch_".
            final_output_name (str, optional): Filename for merged result. Defaults to "merged_result.json".
        """
        self.ai_model = ai_model
        self.max_tokens = max_tokens
        self.buffer_tokens = buffer_tokens
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.final_output_name = final_output_name
        
        # Initialize tokenizer for token counting
        self.enc = tiktoken.encoding_for_model(ai_model)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.enc.encode(text))
    
    def dynamic_batch_generator(
        self, 
        df: pd.DataFrame, 
        columns_for_tokenization: List[str], 
        row_formatter: Callable[[Dict], str]
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Generate batches of DataFrame rows based on token limits.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns_for_tokenization (List[str]): Columns containing text to be measured for tokens
            row_formatter (Callable): Function that formats a row dict into a string for token counting
            
        Yields:
            pd.DataFrame: Batch of rows
        """
        batch, batch_tokens = [], 0

        for _, row in df.iterrows():
            row_text = row_formatter(row.to_dict())
            row_tokens = self.count_tokens(row_text)

            # Check if adding this row would exceed the token limit
            if batch_tokens + row_tokens > self.max_tokens - self.buffer_tokens:
                if batch:  # Only yield if we have something in the batch
                    yield pd.DataFrame(batch)
                batch, batch_tokens = [], 0

            # Add current row to batch
            batch.append(row.to_dict())
            batch_tokens += row_tokens

        # Yield any remaining rows
        if batch:
            yield pd.DataFrame(batch)

    async def call_agent(
        self, 
        agent: ChatCompletionAgent, 
        user_input: str,
        settings: Optional[PromptExecutionSettings] = None
    ) -> Optional[str]:
        """
        Stream the response from the provided agent and return the full concatenated string.
        
        Args:
            agent (ChatCompletionAgent): The agent to invoke
            user_input (str): The user input to process
            settings (Optional[PromptExecutionSettings]): Optional execution settings
            
        Returns:
            Optional[str]: The complete model response or None if error
        """
        history = ChatHistory()
        history.add_user_message(user_input)

        kwargs = {}
        if settings:
            kwargs["arguments"] = KernelArguments(settings=settings)

        full_response = ""
        try:
            async for part in agent.invoke_stream(history, **kwargs):
                if getattr(part, "content", "").strip():
                    full_response += part.content
        except Exception as e:
            print(f"Agent invocation error: {e}")
            return None
        return full_response

    def extract_json_from_response(self, response: str) -> List[Dict]:
        """
        Extract JSON data from model response text.
        
        Args:
            response (str): The raw model response
            
        Returns:
            List[Dict]: Extracted JSON data as Python objects
        """
        match = re.search(r"\[\s*{[\s\S]*?}\s*\]", response)
        if not match:
            print("No JSON array found in response.")
            return []
            
        try:
            data = json.loads(match.group(0).strip())
            return data if isinstance(data, list) else [data]
        except Exception as e:
            print(f"JSON parse error: {e}")
            return []

    def save_raw_output(self, response: str, batch_idx: int) -> str:
        """
        Save raw model output to disk.
        
        Args:
            response (str): The raw model response
            batch_idx (int): Batch index number
            
        Returns:
            str: Path to the saved file
        """
        path = os.path.join(self.output_dir, f"{self.output_prefix}{batch_idx}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"Raw output saved: {path}")
        return path

    def merge_batch_results(self, batches: List[List[Dict]]) -> List[Dict]:
        """
        Merge results from multiple batches, removing duplicates.
        
        Args:
            batches (List[List[Dict]]): List of batch results
            
        Returns:
            List[Dict]: Merged unique results
        """
        seen, merged = set(), []
        for batch in batches:
            for item in batch:
                # Use JSON string as hash for deduplication
                item_sig = json.dumps(item, sort_keys=True)
                if item_sig not in seen:
                    seen.add(item_sig)
                    merged.append(item)
        return merged

    def save_merged_results(self, results: List[Dict]) -> str:
        """
        Save merged results to JSON file.
        
        Args:
            results (List[Dict]): The merged results to save
            
        Returns:
            str: Path to the saved file
        """
        output_path = os.path.join(self.output_dir, self.final_output_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Final merged results saved to {output_path}")
        return output_path

    async def process_batches(
        self,
        df: pd.DataFrame,
        agent: ChatCompletionAgent,
        columns_for_tokenization: List[str],
        row_formatter: Callable[[Dict], str],
        batch_formatter: Callable[[pd.DataFrame], str],
        execution_settings: Optional[PromptExecutionSettings] = None
    ) -> List[Dict]:
        """
        Process a DataFrame through batches using a provided agent and extract structured data.
        
        Args:
            df (pd.DataFrame): Input DataFrame to process
            agent (ChatCompletionAgent): Pre-configured agent to use for processing
            columns_for_tokenization (List[str]): Column names containing text to measure for batching
            row_formatter (Callable): Function to format a single row for token counting
            batch_formatter (Callable): Function to format an entire batch for agent input
            execution_settings (Optional[PromptExecutionSettings]): Optional settings to override agent defaults
            
        Returns:
            List[Dict]: Extracted and merged structured data from all batches
        """
        all_result_batches = []
        
        # Process each batch
        for idx, batch_df in enumerate(self.dynamic_batch_generator(df, columns_for_tokenization, row_formatter), start=1):
            print(f"\n---- Processing batch {idx} ----")
            batch_input = batch_formatter(batch_df)
            
            # Get token count for logging
            token_count = self.count_tokens(batch_input)
            print(f"Batch {idx} size: {len(batch_df)} rows, estimated tokens: {token_count}")
            
            # Call the agent
            raw_response = await self.call_agent(agent, batch_input, execution_settings)
            if not raw_response:
                print(f"Batch {idx}: No response received")
                continue
                
            # Save raw output for debugging
            self.save_raw_output(raw_response, idx)
            
            # Extract JSON data from response
            batch_results = self.extract_json_from_response(raw_response)
            print(f"Batch {idx}: extracted {len(batch_results)} data items")
            
            if batch_results:
                all_result_batches.append(batch_results)
        
        # Merge all batch results
        if not all_result_batches:
            print("No data extracted.")
            return []
            
        merged_results = self.merge_batch_results(all_result_batches)
        self.save_merged_results(merged_results)
        
        return merged_results
        """
        Process a DataFrame through batches using a provided agent and extract structured data.
        
        Args:
            df (pd.DataFrame): Input DataFrame to process
            agent (ChatCompletionAgent): Pre-configured agent to use for processing
            text_columns (List[str]): Column names containing text to measure for batching
            row_formatter (Callable): Function to format a single row for token counting
            batch_formatter (Callable): Function to format an entire batch for agent input
            execution_settings (Optional[PromptExecutionSettings]): Optional settings to override agent defaults
            
        Returns:
            List[Dict]: Extracted and merged structured data from all batches
        """
        all_result_batches = []
        
        # Process each batch
        for idx, batch_df in enumerate(self.dynamic_batch_generator(df, text_columns, row_formatter), start=1):
            print(f"\n---- Processing batch {idx} ----")
            batch_input = batch_formatter(batch_df)
            
            # Get token count for logging
            token_count = self.count_tokens(batch_input)
            print(f"Batch {idx} size: {len(batch_df)} rows, estimated tokens: {token_count}")
            
            # Call the agent
            raw_response = await self.call_agent(agent, batch_input, execution_settings)
            if not raw_response:
                print(f"Batch {idx}: No response received")
                continue
                
            # Save raw output for debugging
            self.save_raw_output(raw_response, idx)
            
            # Extract JSON data from response
            batch_results = self.extract_json_from_response(raw_response)
            print(f"Batch {idx}: extracted {len(batch_results)} data items")
            
            if batch_results:
                all_result_batches.append(batch_results)
        
        # Merge all batch results
        if not all_result_batches:
            print("No data extracted.")
            return []
            
        merged_results = self.merge_batch_results(all_result_batches)
        self.save_merged_results(merged_results)
        
        return merged_results