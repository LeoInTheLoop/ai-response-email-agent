# do it later

# 🔁 索引设计建议
# 你可以为项目和人物设置双向索引：

# 每个项目记录相关人物列表

# 每个人记录参与项目列表

# 建议统一用 email_id 或 conversation_id 做连接锚点

# 🔧 接下来的步骤建议：
# 拆出两个新的 prompt 模板 + agent class（复用当前结构）

# 统一结构结果保存在：

# /data/project_info/

# /data/person_info/

# 提供一个跨项目 / 人物的检索函数（例如：find_person_projects(email)）


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

# Load environment variables
print("Loading environment...")
load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
assert GITHUB_TOKEN, "Please set your GITHUB_TOKEN environment variable"
AI_MODEL = "gpt-4o-mini"

# JSON Template for project extraction
project_extract_format = [
  {
    "project_name": "brief name for the project",
    "description": "concise summary of the project",
    "stakeholders": ["person1", "person2"],
    "milestones": ["kickoff", "testing", "launch"],
    "timeline": {
      "start": "2025-05-01",
      "end": "2025-06-15"
    },
    "related_emails": ["email_id1", "email_id2"]
  }
]

INSTRUCTIONS_PROJECT = """
You are a project extraction assistant.

Given a group of emails, extract any projects discussed. For each project, output:
- A short name for the project
- A brief description (1–3 lines)
- Stakeholders (email names mentioned or implied)
- Milestones or deliverables mentioned
- Approximate timeline (if dates mentioned)
- Email identifiers or descriptions

ONLY output a JSON array of project objects like this:
{project_extract_format}
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
service_id = "github-project-agent"
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

async def extract_projects(df):
    print("Starting extract_projects()")
    BATCH_SIZE = 5
    all_projects = []

    for idx, batch_df in enumerate(batch_email_generator(df, BATCH_SIZE)):
        print(f"\n---- Processing batch {idx+1} ----")
        email_text = get_email_summary_text(batch_df)
        print(f"Batch {idx+1}, text length: {len(email_text)}")

        instruction = INSTRUCTIONS_PROJECT.format(
            project_extract_format=json.dumps(project_extract_format, ensure_ascii=False, indent=2)
        )
        agent = ChatCompletionAgent(
            kernel=kernel,
            name="ProjectExtractAgent",
            instructions=instruction,
            arguments=KernelArguments(settings=PROMPT_SETTING)
        )

        full_response, err = await process_one_batch(agent, email_text)
        if not full_response:
            print(f"Batch {idx+1}: Error or empty response:", err)
            continue

        # Save raw output for inspection
        with open(f"../../data/project_info/projects_batch_{idx+1}_raw.txt", "w", encoding="utf-8") as f:
            f.write(full_response)

        try:
            batch_json = json.loads(full_response.strip())
            if not isinstance(batch_json, list):
                batch_json = [batch_json]
            print(f"Batch {idx+1}: JSON loaded with {len(batch_json)} projects")
            all_projects.extend(batch_json)
        except Exception as e:
            print(f"Batch {idx+1}: Failed to parse JSON. Skipping. Exception: {e}")

    # Deduplicate by string hash
    seen = set()
    unique_projects = []
    for proj in all_projects:
        proj_str = json.dumps(proj, sort_keys=True)
        if proj_str not in seen:
            seen.add(proj_str)
            unique_projects.append(proj)

    if unique_projects:
        with open("../../data/project_info/projects_summary_total.json", "w", encoding="utf-8") as f:
            json.dump(unique_projects, f, ensure_ascii=False, indent=2)
        print("Final project summary saved as projects_summary_total.json")
    else:
        print("No projects extracted.")

# Entry point
if __name__ == "__main__":
    MAX_EMAIL_PROCESS = 30
    csv_path = "../../data/train_dataset/phillip_allen_emails.csv"

    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path)
    df = df.head(MAX_EMAIL_PROCESS)
    print("CSV loaded, shape:", df.shape)

    asyncio.run(extract_projects(df))
