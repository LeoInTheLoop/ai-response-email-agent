import os
import json
import pandas as pd
import asyncio
from dotenv import load_dotenv

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from openai import AsyncOpenAI

from utils_email import get_email_summary_text
import re

# Load environment variables
print("Loading environment...")
load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
assert GITHUB_TOKEN, "Please set your GITHUB_TOKEN environment variable"
AI_MODEL = "gpt-4o-mini"

# Initialize JSON format template
# Initialize JSON format template with a serializable placeholder
extract_format = [
  {
    "context": "describe the typical situation where this style is used",
    "tone": "summarize the tone, e.g., formal / friendly / direct / humorous",
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

# Define instruction template
INSTRUCTIONS_BASE = """
You are an email style analysis assistant.

Given several emails written by the same user, your task is to identify different writing styles the user tends to use, if any. These styles might vary by context — for example: talking to friends, clients, colleagues, or replying quickly on mobile.

For each style you find, output the following as a JSON object:

- context: what kind of situation this style is used in (you can summarize from the email thread or recipient)
- tone: a brief description of tone (e.g. friendly, concise, formal)
- greeting: typical greeting phrases used, not names
- closing: common closing lines
- patterns: sentence patterns or structures the user repeats
- keywords: specific words or phrases the user tends to use
- signature: how the user usually ends the email

Output a JSON array of such objects. If the user has only one general style, return just one object in the array.

Do not invent content. Only summarize what you observe from the user's actual emails.

Here are the user's emails:
===
{user_email}
===
Format is :
{extract_format}
"""

PROMPT_SETTING = PromptExecutionSettings(
    temperature=0.2,
    top_p=0.9,
    max_tokens=4096
)

print("Creating AsyncOpenAI client")
client = AsyncOpenAI(
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com/"
)
print("AsyncOpenAI client created")

print("Initializing Kernel and chat service")
kernel = Kernel()
service_id = "github-agent"
chat_service = OpenAIChatCompletion(
    ai_model_id=AI_MODEL,
    async_client=client,
    service_id=service_id
)
kernel.add_service(chat_service)

def batch_email_generator(df, batch_size):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size]

async def process_one_batch(agent, user_input):
    chat_history = ChatHistory()
    chat_history.add_user_message(user_input)
    full_response = ""
    try:
        async for content in agent.invoke_stream(chat_history):
            if hasattr(content, 'content') and content.content.strip():
                full_response += content.content
    except Exception as e:
        print("Error during agent.invoke_stream:", e)
        return None, str(e)
    return full_response, None

async def analyze_emails(df, user_email):
    print("Starting analyze_emails()")
    BATCH_SIZE = 5
    all_summaries = []

    for idx, batch_df in enumerate(batch_email_generator(df, BATCH_SIZE)):
        print(f"\n---- Processing batch {idx+1} ----")
        email_history_text = get_email_summary_text(batch_df)
        print(f"Batch {idx+1}, history length: {len(email_history_text)}")

        batch_instructions = INSTRUCTIONS_BASE.format(
            user_email=user_email,
            extract_format=json.dumps(extract_format, ensure_ascii=False, indent=2)
        )

        agent = ChatCompletionAgent(
            kernel=kernel,
            name="ExtractAgent",
            instructions=batch_instructions,
            arguments=KernelArguments(settings=PROMPT_SETTING)
        )

        full_response, err = await process_one_batch(agent, email_history_text)
        if not full_response:
            print(f"Batch {idx+1}: Error or empty response:", err)
            continue

        # Save raw AI output for debugging
        raw_path = f"../../data/summary_data/email_style_projects_summary_batch_{idx+1}_raw.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(full_response)
        print(f"Saved raw output to {raw_path}")

        try:
            # Clean up and extract JSON block from response
            json_match = re.search(r"\[.*\]", full_response, re.DOTALL)
            if json_match:
                json_text = json_match.group(0).strip()
                batch_json = json.loads(json_text)
                if not isinstance(batch_json, list):
                    batch_json = [batch_json]
                print(f"Batch {idx+1}: Parsed {len(batch_json)} style(s)")
                all_summaries.append(batch_json)
            else:
                print(f"Batch {idx+1}: No JSON array found in output.")
        except json.JSONDecodeError as e:
            print(f"Batch {idx+1}: JSON decode error:", e)
        except Exception as e:
            print(f"Batch {idx+1}: Unexpected error during JSON parsing:", e)

    def merge_summaries(summaries_list):
        all_styles = []
        seen = set()
        for batch in summaries_list:
            for style in batch:
                style_str = json.dumps(style, sort_keys=True)
                if style_str not in seen:
                    seen.add(style_str)
                    all_styles.append(style)
        return all_styles

    if all_summaries:
        merged = merge_summaries(all_summaries)
        output_path = "../../data/summary_data/email_style_projects_summary_total.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Final merged summary saved as {output_path}")
    else:
        print("No summary generated.")
# Entry point for testing
if __name__ == "__main__":
    user_email = "phillip.allen@enron.com"
    MAX_EMAIL_PROCESS = 20
    csv_path = "../../data/train_dataset/phillip_allen_emails.csv"

    # Read and preprocess CSV
    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path)
    df = df.head(MAX_EMAIL_PROCESS) if MAX_EMAIL_PROCESS > 0 else df
    print("CSV loaded, shape:", df.shape)

    asyncio.run(analyze_emails(df, user_email))