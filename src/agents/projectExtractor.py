# do it later
# milestones.due
# [
#   {
#     "project": "AI Email Agent",
#     "description": "A smart email assistant that analyzes tone and generates replies.",
#     "status": "active",
#     "start_date": "2025-04-18",
#     "end_date": "2025-06-01",
#     "milestones": [
#       {
#         "name": "Tone analysis module",
#         "due": "2025-04-25",
#         "completed": true,
#         "contributors": ["Leo", "Chen"],
#         "notes": "Tested on 50 samples."
#       },
#       {
#         "name": "Reply generation agent",
#         "due": "2025-05-05",
#         "completed": true,
#         "contributors": ["Leo"],
#         "notes": "Integrated GPT-4o-mini with plugin calling."
#       },
#       {
#         "name": "Project extractor",
#         "due": "2025-05-15",
#         "completed": false,
#         "contributors": ["Leo", "Anna"],
#         "notes": "Currently extracting project + person info from emails."
#       }
#     ],
#     "members": [
#       {
#         "name": "Leo",
#         "role": "Lead developer",
#         "focus": ["backend", "agent logic"],
#         "achievements": ["Built reply generator", "Designed plugin system"]
#       },
#       {
#         "name": "Anna",
#         "role": "NLP engineer",
#         "focus": ["entity extraction", "summarization"],
#         "achievements": ["Email entity extraction pipeline"]
#       },
#       {
#         "name": "Chen",
#         "role": "UX/UI",
#         "focus": ["prompt interface", "frontend support"]
#       }
#     ],
#     "tools": ["GPT-4o", "Semantic Kernel", "FastAPI"],
#     "keywords": ["email", "tone", "project management", "agent"]
#   }
# ]


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
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

from config import DATA_DIR, RAW_EMAILS_PATH, TRAIN_DATA_PATH, TEMPLATE_DIR, LOG_DIR

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



# Output template
# Output template for project extractor
extract_format = [
    {
        "project": "Project Name",
        "description": "Short description of the project",
        "status": "active / completed / paused / unknown",
        "start_date": "YYYY-MM-DD or unknown",
        "end_date": "YYYY-MM-DD or unknown",
        "milestones": [
            {
                "name": "Milestone name",
                "due": "YYYY-MM-DD or unknown",
                "completed": True,
                "contributors": ["Name1", "Name2"],
                "notes": "Optional note or summary"
            }
        ],
        "members": [
            {
                "name": "Member name",
                "role": "e.g., Developer / Researcher / Manager",
                "focus": ["Main focus areas"],
                "achievements": ["Optional key contributions"]
            }
        ],
        "keyword": ["words people use to refer to this project"],
        "lastupdate": "YYYY-MM-DD"
    }
]

INSTRUCTIONS_BASE = """
You are a project information extractor.

Given several emails, your task is to identify any ongoing or past projects mentioned in them. For each project, summarize the following:

- project name
- description
- current status (active / completed / paused / unknown)
- start and end dates (if known)
- milestones (name, due date, contributors, completion status, notes)
- members (name, role, focus areas, notable achievements)
- keywords people use to refer to this project
- last update time you can infer from the emails

Output the result as a JSON array. Only extract information that can be inferred directly from the email content — do not make up facts.

Here are the emails:
===
{emails_block}
===
Format:
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

    processor = JsonBatchProcessor(output_dir=DATA_DIR)

    all_style_batches: list[list[dict]] = []

    for idx, batch_df in enumerate(
        processor.dynamic_batch_generator(df, ["subject", "body"], format_row), start=1
    ):
        print(f"\n---- Processing batch {idx} ----")

        batch_text = df_to_text(batch_df)
        prompt = format_batch(batch_df)

        token_estimate = processor.count_tokens(prompt) + processor.count_tokens(batch_text)
        print(f"Batch {idx} size: {len(batch_df)} emails, estimated tokens: {token_estimate}")
        raw_response = ""
        if token_estimate > 8000:
            print(f"Batch {idx} exceeds max tokens ({8000}). half")
            half_size = len(batch_df) // 2
            batch_df1, batch_df = batch_df.iloc[:half_size], batch_df.iloc[half_size:]
            batch_text1,batch_text = df_to_text(batch_df1), df_to_text(batch_df)
            prompt1, prompt  = format_batch(batch_df1), format_batch(batch_df)
            raw_response1 = await processor.call_model(
                kernel,
                prompt=prompt1,
                user_input=batch_text1,
                execution_settings=PROMPT_SETTING
            )
            raw_response += raw_response1 if raw_response1 else ""
              

        raw_response2 =  await processor.call_model(
            kernel,
            prompt=prompt,
            user_input=batch_text,
            execution_settings=PROMPT_SETTING
        )
        raw_response += raw_response2 if raw_response2 else ""
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

    merged_path = os.path.join(DATA_DIR, "project_summary_total.json")
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_styles, f, ensure_ascii=False, indent=2)
    print(f"Final merged summary saved to {merged_path}")

    return merged_styles

# CLI test
if __name__ == "__main__":
    user_email = "phillip.allen@enron.com"
    MAX_EMAIL_PROCESS = 30
    csv_path = TRAIN_DATA_PATH

    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path).head(MAX_EMAIL_PROCESS)
    print("CSV loaded, shape:", df.shape)

    asyncio.run(analyze_emails(df, user_email))
