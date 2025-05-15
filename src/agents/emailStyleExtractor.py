import os
import json
import pandas as pd
import asyncio
from dotenv import load_dotenv

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

# Import reusable functions and batch processor
from agents.utils.create_kernel_and_agent import (
    create_kernel,
    add_chat_service,
    DEFAULT_AI_MODEL
)
from agents.utils.JsonBatchProcessor import JsonBatchProcessor

# Load environment variables
print("Loading environment...")
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
assert GITHUB_TOKEN, "Please set your GITHUB_TOKEN environment variable"

# Filepaths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(base_dir, "../../data/summary_data"))

# Output template
extract_format = [
    {
        "context": "describe the typical situation where this style is used",
        "tone": ["list summarize the tone, e.g., formal / friendly / direct / humorous"],
        "actor": ["list who the email is addressed to, e.g., Client / Manager / Colleague / Friend"],
        "intent": ["list what the email is trying to achieve, e.g., Technical support / Coordination / Request / Follow-up"],
        "greeting": ["example greeting phrase 1", "example greeting phrase 2"],
        "closing": ["example closing phrase 1", "example closing phrase 2"],
        "patterns": ["example sentence structure or phrase often used"],
        "keywords": ["frequently used words or expressions"],
        "signature": "typical way the user signs off"
    }
]

# Prompt template
INSTRUCTIONS_BASE = """
You are an email style analysis assistant.

Given several emails written by the same user, your task is to identify different writing styles the user tends to use, if any. These styles might vary by context — for example: talking to friends, clients, colleagues, or replying quickly on mobile.

For each style you find, output the following as a JSON object:

- context
- tone
- actor
- intent
- greeting
- closing
- patterns
- keywords
- signature

Output a JSON array of such objects. If the user has only one general style, return just one object in the array.

Do not invent content. Only summarize what you observe from the user's actual emails.

Here are the user's emails:
===
{emails_block}
===
Format is:
{format_json}
"""

PROMPT_SETTING = PromptExecutionSettings(
    temperature=0.2,
    top_p=0.9,
    max_tokens=4096
)

# Utilities
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

def format_row(row: dict) -> str:
    subj = str(row.get("subject", "")).strip()
    body = str(row.get("body", "")).strip()
    return f"Subject: {subj}\nBody:\n{body}\n"

def format_batch(batch_df: pd.DataFrame) -> str:
    batch_text = df_to_text(batch_df)
    return INSTRUCTIONS_BASE.format(
        emails_block=batch_text,
        format_json=json.dumps(extract_format, ensure_ascii=False, indent=2)
    )

# Main logic
async def analyze_emails(
    df: pd.DataFrame,
    user_email: str,
    batch_size: int | None = None
):
    print("Starting analyze_emails()")

    kernel = create_kernel()
    add_chat_service(kernel, service_id="github-agent")

    processor = JsonBatchProcessor(output_dir=data_dir)

    all_style_batches: list[list[dict]] = []

    for idx, batch_df in enumerate(
        processor.dynamic_batch_generator(df, ["subject", "body"], format_row), start=1
    ):
        print(f"\n---- Processing batch {idx} ----")

        batch_text = df_to_text(batch_df)
        prompt = format_batch(batch_df)

        token_estimate = processor.count_tokens(prompt) + processor.count_tokens(batch_text)
        print(f"Batch {idx} size: {len(batch_df)} emails, estimated tokens: {token_estimate}")

        raw_response = await processor.call_model(
            kernel,
            prompt=prompt,
            user_input=batch_text,
            execution_settings=PROMPT_SETTING
        )
        if not raw_response:
            print(f"Batch {idx} returned no response.")
            continue

        processor.save_raw_output(raw_response, idx)

        styles = processor.extract_json_from_response(raw_response)
        print(f"Batch {idx}: extracted {len(styles)} style(s)")

        if styles:
            all_style_batches.append(styles)

    if not all_style_batches:
        print("No styles extracted.")
        return None

    merged_styles = processor.merge_batch_results(all_style_batches)

    merged_path = os.path.join(data_dir, "tone_summary_total.json")
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_styles, f, ensure_ascii=False, indent=2)
    print(f"Final merged summary saved to {merged_path}")

    return merged_styles

# CLI test
if __name__ == "__main__":
    user_email = "phillip.allen@enron.com"
    MAX_EMAIL_PROCESS = 30
    csv_path = os.path.normpath(os.path.join(base_dir, "../../data/train_dataset/phillip_allen_emails.csv"))

    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path).head(MAX_EMAIL_PROCESS)
    print("CSV loaded, shape:", df.shape)

    asyncio.run(analyze_emails(df, user_email))
