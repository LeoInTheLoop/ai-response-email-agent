###############################################################################
# ------------------------------  IMPORTS  ---------------------------------- #
###############################################################################
import os, json, re, asyncio
import pandas as pd
from dotenv import load_dotenv

from semantic_kernel.kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

# Import reusable functions
from agents.utils.create_kernel_and_agent import (
    create_kernel,
    add_chat_service,
    create_agent,
    DEFAULT_AI_MODEL,
    DEFAULT_PROMPT_SETTINGS
)

###############################################################################
# ------------------------------  CONSTANTS  -------------------------------- #
###############################################################################

# Load environment variables
print("Loading environment...")
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
assert GITHUB_TOKEN, "Please set your GITHUB_TOKEN environment variable"

# Define model token limits
MODEL_TOKEN_LIMITS = {
    "gpt-4o-mini": 8000,  # Example limit
    "gpt-4o": 8000,
}

# filepath
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(base_dir, "../../data/summary_data"))

###############################################################################
# -----------------------  JSON OUTPUT TEMPLATE  ---------------------------- #
###############################################################################
extract_format = [
    {
        "context": "describe the typical situation where this style is used",
        "tone": "summarize the tone, e.g., formal / friendly / direct / humorous",
        "actor": "who the email is addressed to, e.g., Client / Manager / Colleague / Friend",
        "intent": "what the email is trying to achieve, e.g., Technical support / Coordination / Request / Follow-up",
        "greeting": [
            "example greeting phrase 1",
            "example greeting phrase 2"
        ],
        "closing": [
            "example closing phrase 1",
            "example closing phrase 2"
        ],
        "patterns": [
            "example sentence structure or phrase often used"
        ],
        "keywords": [
            "frequently used words or expressions"
        ],
        "signature": "typical way the user signs off"
    }
]

###############################################################################
# --------------------------  PROMPT TEMPLATE  ------------------------------ #
###############################################################################
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
    max_tokens=4096        # generation length (completion), not the context limit
)

###############################################################################
# -----------------------------  UTILITIES  --------------------------------- #
###############################################################################
def df_to_text(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame batch to plain text:
        --- Email 1 ---
        Subject: ...
        Body:
        ...
    The model sees the raw subject + body exactly as written.
    """
    lines: list[str] = []
    for idx, row in df.iterrows():
        subj = str(row.get("subject", "")).strip()
        body = str(row.get("body", "")).strip()
        lines.append(f"--- Email {len(lines) + 1} ---")
        lines.append(f"Subject: {subj}")
        lines.append("Body:")
        lines.append(body)
        lines.append("")  # blank line for spacing
    return "\n".join(lines)

import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def dynamic_batch_generator(df: pd.DataFrame, max_tokens: int = 8000, buffer_tokens: int = 3500):
    batch, batch_tokens = [], 0

    for _, row in df.iterrows():
        subj = str(row.get("subject", "")).strip()
        body = str(row.get("body", "")).strip()
        email_text = f"Subject: {subj}\nBody:\n{body}\n"
        email_tokens = count_tokens(email_text)

        if batch_tokens + email_tokens > max_tokens - buffer_tokens:
            yield pd.DataFrame(batch)
            batch, batch_tokens = [], 0

        batch.append(row.to_dict())
        batch_tokens += email_tokens

    if batch:
        yield pd.DataFrame(batch)

###############################################################################
# -----------------------  SEMANTIC KERNEL HELPERS  ------------------------- #
###############################################################################
def build_prompt(batch_text: str) -> str:
    """Inject email block + format JSON into the system prompt."""
    return INSTRUCTIONS_BASE.format(
        emails_block=batch_text,
        format_json=json.dumps(extract_format, ensure_ascii=False, indent=2)
    )


async def call_model(kernel: Kernel, prompt: str, user_input: str) -> str | None:
    """
    Stream the response from the model and return the full concatenated string.
    """
    agent = create_agent(
        kernel=kernel,
        instructions=prompt,
        service_id="github-agent",
        agent_name="ExtractAgent",
        settings=PROMPT_SETTING
    )
    
    history = ChatHistory()
    history.add_user_message(user_input)

    full_response = ""
    try:
        async for part in agent.invoke_stream(history):
            if getattr(part, "content", "").strip():
                full_response += part.content
    except Exception as e:
        print("invoke_stream error:", e)
        return None
    return full_response


def save_raw_output(response: str, idx: int):
    """Write raw model output to disk for debugging."""
    path = os.path.join(data_dir, f"email_style_batch_{idx}.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(response)
    print(f"Raw output saved: {path}")


def extract_json_from_response(resp: str) -> list[dict]:
    """
    Pull the first JSON array found in the model output and return it as Python
    data. If nothing parsable found, return an empty list.
    """
    match = re.search(r"\[\s*{[\s\S]*?}\s*\]", resp)
    if not match:
        print("No JSON array found.")
        return []
    try:
        data = json.loads(match.group(0).strip())
        return data if isinstance(data, list) else [data]
    except Exception as e:
        print("JSON parse error:", e)
        return []


def merge_batches(batches: list[list[dict]]) -> list[dict]:
    """
    Remove duplicate style objects across batches (dedup by JSON string).
    """
    seen, merged = set(), []
    for batch in batches:
        for style in batch:
            sig = json.dumps(style, sort_keys=True)
            if sig not in seen:
                seen.add(sig)
                merged.append(style)
    return merged


###############################################################################
# ---------------------------  MAIN ENTRYPOINT  ----------------------------- #
###############################################################################
async def analyze_emails(
    df: pd.DataFrame,
    user_email: str,
    batch_size: int | None = None
):
    """
    1) For each batch: build prompt → call model → parse JSON,
    2) Merge all style objects and write to disk.
    """
    print("Starting analyze_emails()")

    kernel = create_kernel()
    add_chat_service(kernel, service_id="github-agent")
    
    all_style_batches: list[list[dict]] = []

    for idx, batch_df in enumerate(dynamic_batch_generator(df,7000), start=1):
        print(f"\n---- Processing batch {idx} ----")
        batch_text = df_to_text(batch_df)
        prompt = build_prompt(batch_text)
        print(f"Batch {idx} size: {len(batch_df)} emails, estimated tokens: {count_tokens(prompt) + count_tokens(batch_text)}")
        raw_response = await call_model(kernel, prompt, batch_text)
        if not raw_response:
            continue

        save_raw_output(raw_response, idx)
        styles = extract_json_from_response(raw_response)
        print(f"Batch {idx}: extracted {len(styles)} style(s)")
        if styles:
            all_style_batches.append(styles)

    if not all_style_batches:
        print("No styles extracted.")
        return None

    merged_styles = merge_batches(all_style_batches)

    merged_path = os.path.join(data_dir, "tone_summary_total.json")
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_styles, f, ensure_ascii=False, indent=2)
    print(f"Final merged summary saved to {merged_path}")

    return merged_styles


###############################################################################
# -------------------------------  CLI TEST  -------------------------------- #
###############################################################################
if __name__ == "__main__":
    user_email = "phillip.allen@enron.com"
    MAX_EMAIL_PROCESS = 80
    csv_path = os.path.normpath(
        os.path.join(base_dir, "../../data/train_dataset/phillip_allen_emails.csv")
    )

    # Read and preprocess CSV (for testing)
    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path).head(MAX_EMAIL_PROCESS)
    print("CSV loaded, shape:", df.shape)

    # Run analysis (df includes 'subject' and 'body' columns)
    asyncio.run(analyze_emails(df, user_email))