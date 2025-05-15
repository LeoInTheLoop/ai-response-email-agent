import os
import json
import pandas as pd
import asyncio
from dotenv import load_dotenv

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory

# Import reusable functions and batch processor
from agents.utils.create_kernel_and_agent import (
    create_agent,
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
extract_format = {
  "context": "describe the typical situation where this style is used",
  "tone": {
    "Vendor": ["Professional", "Firm"]
  },
  "intent": "Negotiation / Issue Resolution",
  "example": "Discussing contract terms",
  "greeting": ["example greeting phrase 1", "example greeting phrase 2"],
  "closing": ["example closing phrase 1", "example closing phrase 2"],
  "patterns": ["example sentence structure or phrase often used"],
  "keywords": ["frequently used words or expressions"],
  "signature": "typical way the user signs off"
}


# Prompt template
INSTRUCTIONS_BASE = """
You are an email style analysis assistant.

Given several emails written by the same user, your task is to identify different writing styles the user tends to use, if any. These styles might vary by context — for example: talking to friends, clients, colleagues, or replying quickly on mobile.

For each distinct style you identify, output the following fields as a JSON object:

- "context": A brief description of when this style is typically used.
- "tone": An object mapping the role to a list of tones, e.g., {{"Client": ["Formal", "Polite"]}}. Choose only one role per style from the list below.
- "intent": The main communicative intent of this style (see options below).
- "example": A typical scenario where this style is used (see options below).
- "greeting": A list of greeting phrases commonly used in this style.
- "closing": A list of closing phrases used in this style.
- "patterns": Common sentence structures or expressions used.
- "keywords": Frequently used words or technical terms.
- "signature": How the user typically signs off.

Only use content you observe directly from the emails. Do not make up patterns or tones. If the user shows only one general style, return a single item in the JSON array.

You must choose one of the following predefined tone combinations (role + tone + intent + example):

1. {{"Client": ["Formal", "Technical"]}}
   - Intent: "Clarify / Request Info"
   - Example: "Explaining investment structures"

2. {{"Client": ["Formal", "Polite"]}}
   - Intent: "Follow-up / Update"
   - Example: "Sending project status reports"

3. {{"Manager": ["Formal", "Direct"]}}
   - Intent: "Request Approval"
   - Example: "Asking for budget sign-off"

4. {{"Manager": ["Concise", "Data-driven"]}}
   - Intent: "Provide Updates"
   - Example: "Summarizing quarterly results"

5. {{"Colleague": ["Friendly", "Direct"]}}
   - Intent: "Coordination"
   - Example: "Scheduling a team meeting"

6. {{"Colleague": ["Casual", "Collaborative"]}}
   - Intent: "Brainstorming"
   - Example: "Discussing project ideas"

7. {{"Investor": ["Formal", "Persuasive"]}}
   - Intent: "Proposal / Pitch"
   - Example: "Presenting a new funding opportunity"

8. {{"Investor": ["Transparent", "Detailed"]}}
   - Intent: "Financial Reporting"
   - Example: "Sharing fiscal performance"

9. {{"Friend": ["Humorous", "Casual"]}}
   - Intent: "Personal Request"
   - Example: "Asking for a favor informally"

10. {{"Friend": ["Warm", "Supportive"]}}
    - Intent: "Check-in / Catch-up"
    - Example: "Following up on personal news"

11. {{"Vendor": ["Professional", "Firm"]}}
    - Intent: "Negotiation / Issue Resolution"
    - Example: "Discussing contract terms"

12. {{"Vendor": ["Polite", "Urgent"]}}
    - Intent: "Logistics Follow-up"
    - Example: "Tracking a delayed shipment"

13. {{"General": ["Neutral", "Polite"]}}
    - Intent: "General Purpose"
    - Example: "Sharing general updates or announcements"

Use only these combinations. Do not invent new ones.

Output should be a JSON array of style objects, each containing exactly one of the tone combinations above, along with fields based on what you observe from the user's emails.

Format is:
===
{format_json}
===
Here are the user's emails:
===
{emails_block}
===

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

    # 只创建一次 agent
    prompt_template = INSTRUCTIONS_BASE.format(
        emails_block="{placeholder}",  # 预留占位符
        format_json=json.dumps(extract_format, ensure_ascii=False, indent=2)
    )
    agent = create_agent(
        kernel=kernel,
        instructions=prompt_template,
        service_id="github-agent",
        agent_name="ExtractAgent",
        settings=PROMPT_SETTING
    )

    all_style_batches: list[list[dict]] = []

    for idx, batch_df in enumerate(
        processor.dynamic_batch_generator(df, ["subject", "body"], format_row), start=1
    ):
        print(f"\n---- Processing batch {idx} ----")

        emails_text = df_to_text(batch_df)

        history = ChatHistory()
        history.add_user_message(emails_text)

        raw_response = ""
        try:
            async for part in agent.invoke_stream(history):
                if getattr(part, "content", "").strip():
                    raw_response += part.content
        except Exception as e:
            print("invoke_stream error:", e)
            continue

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
