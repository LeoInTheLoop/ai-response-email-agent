import os
import json
import pandas as pd
import asyncio
from dotenv import load_dotenv
import re

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from openai import AsyncOpenAI

print("Loading environment...")
load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
print("GITHUB_TOKEN:", GITHUB_TOKEN is not None)
assert GITHUB_TOKEN, "Please set your GITHUB_TOKEN environment variable"
AI_MODEL = "gpt-4o-mini"  
csv_path = "train_dataset/phillip_allen_emails.csv"
user_email = "phillip.allen@enron.com"
MAX_EMAIL_PROCESS = 20

print("Loading CSV:", csv_path)
df = pd.read_csv(csv_path)
df = df.head(MAX_EMAIL_PROCESS) if MAX_EMAIL_PROCESS > 0 else df
print("CSV loaded, shape:", df.shape)

def get_email_summary_text(df):
    entries = []
    for idx, row in df.iterrows():
        msg = row.get('message', '')
        # use re
        from_ = re.search(r'From:\s*(.*)', msg)
        to = re.search(r'To:\s*(.*)', msg)
        cc = re.search(r'X-cc:\s*(.*)', msg)
        bcc = re.search(r'X-bcc:\s*(.*)', msg)
        subject = re.search(r'Subject:\s*(.*)', msg)
        
        # body
        body_start = msg.find('\n\n')
        body = msg[body_start+2:] if body_start != -1 else ''
        from_ = from_.group(1).strip() if from_ else ''
        to = to.group(1).strip() if to else ''
        cc = cc.group(1).strip() if cc else ''
        bcc = bcc.group(1).strip() if bcc else ''
        subject = subject.group(1).strip() if subject else ''
        entries.append(
            f"""From: {from_}
To: {to}
CC: {cc}
BCC: {bcc}
Subject: {subject}
Body: {body.strip()}
-------------------------------"""
        )
    return "\n".join(entries)

INSTRUCTIONS_BASE = """
Given the following email history involving {user_email} (showing sender, receiver, subject, and body), analyze how the main user (the sender: {user_email}) communicates with each recipient.

For each communication partner (recipient), extract and summarize:
- tone: A mapping where keys are types of tone used by the sender (e.g. "formal", "casual", "concise", "friendly", "assertive", etc.) and values are the corresponding frequent words/phrases that typify each tone (for example: "formal": ["regards", "please advise"], "casual": ["thanks", "see you"]). Please include at least 2-3 tone types if possible.
- style: Description of writing style with this person (examples: short sentences, polite, direct, detailed, analytic, etc.)
- projects: For any project mentioned, provide a list of objects each containing:
    - name: Project name
    - collaborators: List of collaborators (preferably as email addresses)
    - details: Any short description or context about this project.
- characteristic_phrases: List at least 3 typical words or frequent expressions you recognize in these emails with this recipient.
- frequent_words: List the top 5-10 frequent words or expressions, ignoring common stop-words and generic terms.

Return your answer as a valid JSON dictionary in the following format:

{{
  "overall": {{ 
    "tone": {{
      "formal": "...",
      "casual": "...",
      ...
    }},
    "style": "...",
    "projects": [
      {{
        "name": "...",
        "collaborators": ["...", "..."],
        "details": "..."
      }}
    ],
    "characteristic_phrases": ["...", "...", "..."],
    "frequent_words": ["...", "...", "..."]
  }},
  "recipient_email_1": {{
    "tone": {{
      "formal": ["frequent words or phrases..."], 
      "casual": ["frequent words or phrases..."], 
      "concise": ["frequent words or phrases..."],
      ...
    }},
    "style": "...",
    "projects": [
      {{
        "name": "...",
        "collaborators": ["...", "..."],
        "details": "..."
      }}
    ],
    "characteristic_phrases": ["...", "...", "..."],
    "frequent_words": ["...", "...", "..."]
  }},
  ...,

}}

If a field cannot be extracted, set its value to null or an empty list.

You must only return valid JSON. Do not include any additional text outside the JSON object.
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

print("Initializing ChatCompletionAgent")
# --- Loop over batches ---
def batch_email_generator(df, batch_size):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]

def combine_json(summary1, summary2):
    
    if not summary1: return summary2
    if not summary2: return summary1
    result = summary1.copy()
    for k, v in summary2.items():
        if k not in result:
            result[k] = v
        elif isinstance(result[k], list) and isinstance(v, list):
          
            result[k].extend([item for item in v if item not in result[k]])
        elif isinstance(result[k], dict) and isinstance(v, dict):
            for kk, vv in v.items():
                if kk not in result[k]:
                    result[k][kk] = vv
                elif isinstance(result[k][kk], list) and isinstance(vv, list):
                    result[k][kk].extend([item for item in vv if item not in result[k][kk]])
    return result

async def process_one_batch(agent, user_input):
    chat_history = ChatHistory()
    chat_history.add_user_message(user_input)
    full_response = ""
    try:
        async for content in agent.invoke_stream(chat_history):
            if (hasattr(content, 'content') and content.content.strip() and
                not any(isinstance(item, (FunctionCallContent, FunctionResultContent)) for item in content.items)):
                full_response += content.content
    except Exception as e:
        print("Error during agent.invoke_stream:", e)
        return None, str(e)
    return full_response, None

async def main():
    print("Starting main()")
    BATCH_SIZE = 5  # 建议先设小一点确保json不会被截断
    all_summaries = []
    for idx, batch_df in enumerate(batch_email_generator(df, BATCH_SIZE)):
        print(f"\n---- Processing batch {idx+1} ----")
        email_history_text = get_email_summary_text(batch_df)
        print(f"Batch {idx+1}, history length: {len(email_history_text)}")

        batch_instructions = INSTRUCTIONS_BASE.format(user_email=user_email) + \
            "\nNew email history:\n" + email_history_text + \
            "\nReturn the JSON summary only."
        agent = ChatCompletionAgent(
            kernel=kernel,
            name="ExtractAgent",
            instructions=batch_instructions,
            arguments=KernelArguments(
                settings=PROMPT_SETTING
            )
        )

        full_response, err = await process_one_batch(agent, email_history_text)
        if not full_response:
            print(f"Batch {idx+1}: Error or empty response:", err)
            continue

        # 保存每一批原始AI输出
        with open(f"summary_data/email_style_projects_summary_batch_{idx+1}_raw.txt", "w", encoding="utf-8") as f:
            f.write(full_response)

        try:
            batch_json = json.loads(full_response)
            print(f"Batch {idx+1}: JSON loaded successfully")
        except Exception as e:
            print(f"Batch {idx+1}: Failed to parse JSON, skip merge. Exception:", e)
            continue

        all_summaries.append(batch_json)

    # 合并所有批次的 summary
    def merge_summaries(summaries_list):
        result = {}
        for summary in summaries_list:
            for k, v in summary.items():
                if k not in result:
                    result[k] = v
                else:
                    # 合并 recipient 节点
                    for field in ['tone', 'style', 'projects', 'characteristic_phrases', 'frequent_words']:
                        if isinstance(v.get(field), list):
                            if field not in result[k] or not isinstance(result[k][field], list):
                                result[k][field] = []
                            result[k][field].extend([item for item in v[field] if item not in result[k][field]])
                        elif isinstance(v.get(field), dict):
                            if field not in result[k] or not isinstance(result[k][field], dict):
                                result[k][field] = {}
                            result[k][field].update(v[field])
                        elif v.get(field) and (not result[k].get(field)):
                            result[k][field] = v[field]
        return result

    if all_summaries:
        merged = merge_summaries(all_summaries)
        with open("summary_data/email_style_projects_summary_total.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print("Final merged summary saved as email_style_projects_summary_total.json")
    else:
        print("No summary generated.")


if __name__ == "__main__":
    print("Running asyncio main()")
    asyncio.run(main())