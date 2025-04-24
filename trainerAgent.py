# problem description:
#  output limit
#  input upgrade use loop

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

# 1. Clean up email history
print("Loading CSV:", csv_path)
df = pd.read_csv(csv_path)
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

        # deal with None
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

email_history_text = get_email_summary_text(df)
print("Length of history text:", len(email_history_text))
email_history_text_trunc = email_history_text[:12000]  
print("Truncated history length:", len(email_history_text_trunc))
print ("Truncated history text:\n", email_history_text_trunc[:120] )

INSTRUCTIONS = """
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

{
  "overall": { 
    "tone": {
      "formal": "...",
      "casual": "...",
      ...
    },
    "style": "...",
    "projects": [
      {
        "name": "...",
        "collaborators": ["...", "..."],
        "details": "..."
      }
    ],
    "characteristic_phrases": ["...", "...", "..."],
    "frequent_words": ["...", "...", "..."]
  },
  "recipient_email_1": {
    "tone": {
      "formal": ["frequent words or phrases..."], 
      "casual": ["frequent words or phrases..."], 
      "concise": ["frequent words or phrases..."],
      ...
    },
    "style": "...",
    "projects": [
      {
        "name": "...",
        "collaborators": ["...", "..."],
        "details": "..."
      }
    ],
    "characteristic_phrases": ["...", "...", "..."],
    "frequent_words": ["...", "...", "..."]
  },
  ...,

}

If a field cannot be extracted, set its value to null or an empty list.

You must only return valid JSON. Do not include any additional text outside the JSON object.
"""
PROMPT_SETTING = PromptExecutionSettings(
    temperature=0.2,
    top_p=0.9,
    max_tokens=16380
)
USER_INPUT_HISTORY = [email_history_text_trunc]

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
agent = ChatCompletionAgent(
    kernel=kernel,
    name="ExtractAgent",
    instructions=INSTRUCTIONS,
    arguments=KernelArguments(
        settings=PROMPT_SETTING
    )
)

async def main():
    print("Starting main()")
    chat_history = ChatHistory()
    user_input = USER_INPUT_HISTORY[0]
    print("Adding user message to chat history")
    chat_history.add_user_message(user_input)
    full_response = ""
    try:
        print("Invoking agent stream...")
        async for content in agent.invoke_stream(chat_history):
            # print("Received chunk:", content)
            if (hasattr(content, 'content') and content.content and content.content.strip() and
                not any(isinstance(item, (FunctionCallContent, FunctionResultContent)) for item in content.items)):
                full_response += content.content
    except Exception as e:
        print("Error during agent.invoke_stream:", e)
        return
    
    # Output and save

    print("AI raw response:\n", full_response)
    result_filename = None

    try:
        json_obj = json.loads(full_response)
        print("JSON loaded successfully")
        with open("email_style_projects_summary.json", "w", encoding="utf-8") as f:
            json.dump(json_obj, f, ensure_ascii=False, indent=2)
        print("Summary saved as email_style_projects_summary.json")
    except Exception as e:
        print("Failed to parse JSON. Raw content saved as .txt file.")
        print("Exception:", e)
        # 保存原始内容
        with open("email_style_projects_summary_raw.txt", "w", encoding="utf-8") as f:
            f.write(full_response)
        print("Raw response saved as email_style_projects_summary_raw.txt")
if __name__ == "__main__":
    print("Running asyncio main()")
    asyncio.run(main())  