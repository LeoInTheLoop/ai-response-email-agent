import os
import json
import pandas as pd
import asyncio
from dotenv import load_dotenv

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from config import DATA_DIR, TRAIN_DATA_PATH

from agents.utils.create_kernel_and_agent import (
    create_agent,
    create_kernel,
    add_chat_service,
    DEFAULT_AI_MODEL
)
from agents.utils.JsonBatchProcessor import JsonBatchProcessor

print("Loading environment...")
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
assert GITHUB_TOKEN, "Please set your GITHUB_TOKEN environment variable"

# 默认已知项目
existing_projects = [
    {
        "project_id": "proj_001",
        "project_name": "AI Email Agent",
        "project_keywords": ["email", "agent", "GPT", "tone"]
    },
    {
        "project_id": "proj_002",
        "project_name": "Project Extractor",
        "project_keywords": ["extract", "project", "emails"]
    }
]

extract_format = [
    {
        "project_id": "proj_001",
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

- project ID (ususally a unique identifier,and ususally number less than 4, 归类在已有项目，除非必要不新建)
- project name
- description
- current status (active / completed / paused / unknown)
- start and end dates (if known)
- milestones (name, due date, contributors, completion status, notes)
- members (name, role, focus areas, notable achievements)
- keywords people use to refer to this project
- last update time you can infer from the emails

Please note that the following projects are known to you:
{known_projects}

Please prioritize these known projects by matching project names or keywords from the emails. Avoid creating duplicate project entries that represent the same project.

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

def format_known_projects(projects: list[dict]) -> str:
    lines = []
    for p in projects:
        lines.append(f"- Project ID: {p['project_id']}")
        lines.append(f"  Name: {p['project_name']}")
        keywords = ", ".join(p.get("project_keywords", []))
        lines.append(f"  Keywords: {keywords}")
    return "\n".join(lines)

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

def format_batch(batch_df: pd.DataFrame, known_projects: list[dict]) -> str:
    batch_text = df_to_text(batch_df)
    known_proj_text = format_known_projects(known_projects)
    return INSTRUCTIONS_BASE.format(
        emails_block=batch_text,
        format_json=json.dumps(extract_format, ensure_ascii=False, indent=2),
        known_projects=known_proj_text
    )
def update_extracted_projects(projects_batch: list[dict], extracted_projects: dict):
    """
    更新 extracted_projects 字典，
    project_id 作为key,
    合并 keywords（去重汇总）,
    项目名优先保持旧的，如无则更新为新的。
    """
    for proj in projects_batch:
        pid = proj.get("project_id")
        if not pid:
            continue

        new_name = proj.get("project_name", "")
        new_keywords = set(proj.get("project_keywords", []))

        if pid in extracted_projects:
            old_info = extracted_projects[pid]
            existing_name = old_info.get("project_name", "") or new_name
            existing_keywords = set(old_info.get("project_keywords", []))
            merged_keywords = list(existing_keywords.union(new_keywords))
            extracted_projects[pid] = {
                "project_name": existing_name,
                "project_keywords": merged_keywords,
            }
        else:
            extracted_projects[pid] = {
                "project_name": new_name,
                "project_keywords": list(new_keywords),
            }

# ✅ 主函数：项目分析
async def analyze_emails(
    df: pd.DataFrame,
    Recipient_email: str,
    kernel: Kernel,
    known_projects: list[dict] = None,
    batch_size: int | None = None
):
    print("Starting analyze_emails()")
    output_dir = os.path.join(DATA_DIR, "project_info", Recipient_email)
    processor = JsonBatchProcessor(output_dir=output_dir)

    if known_projects is None:
        known_projects = []

    project_dict = {p["project_id"]: p for p in known_projects if "project_id" in p}

    for idx, batch_df in enumerate(
        processor.dynamic_batch_generator(df, ["subject", "body"], format_row), start=1
    ):
        print(f"\n---- Processing batch {idx} ----")

        # ✅ 动态生成 prompt_template
        prompt_template = format_batch(batch_df, known_projects)

        # ✅ 创建 agent 使用 prompt_template
        agent = create_agent(
            kernel=kernel,
            instructions=prompt_template,
            service_id="github-agent",
            agent_name="ExtractAgent",
            settings=PROMPT_SETTING
        )

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

        projects = processor.extract_json_from_response(raw_response)
        print(f"Batch {idx}: extracted {len(projects)} project(s)")

        for proj in projects:
            pid = proj.get("project_id")
            if not pid:
                continue

            if pid not in project_dict:
                project_dict[pid] = proj
            else:
                existing = project_dict[pid]
                existing_keywords = set(existing.get("project_keywords", []))
                new_keywords = set(proj.get("project_keywords", []))
                existing["project_keywords"] = list(existing_keywords.union(new_keywords))

                if not existing.get("project_name") and proj.get("project_name"):
                    existing["project_name"] = proj["project_name"]

    if not project_dict:
        print("No projects extracted.")
        return None

    merged_projects = list(project_dict.values())

    dir_path = os.path.join(DATA_DIR, "project_info")
    os.makedirs(dir_path, exist_ok=True)

    merged_path = os.path.join(dir_path, f"{Recipient_email}_projects.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_projects, f, ensure_ascii=False, indent=2)
    print(f"✅ Final merged summary saved to {merged_path}")

    project_list_path = os.path.join(dir_path, "project_list.json")
    with open(project_list_path, "w", encoding="utf-8") as f:
        json.dump(merged_projects, f, ensure_ascii=False, indent=2)
    print(f"✅ Updated global project list saved to {project_list_path}")

    return merged_projects

# ✅ CLI 测试
if __name__ == "__main__":
    user_email = "phillip.allen@enron.com"
    recipient_email = "stagecoachmama@hotmail.com"
    MAX_EMAIL_PROCESS = 30
    csv_path = TRAIN_DATA_PATH

    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path).head(MAX_EMAIL_PROCESS)
    print("CSV loaded, shape:", df.shape)

    kernel = create_kernel()
    add_chat_service(kernel, service_id="github-agent")

   

    # 加载已知项目
    existing_projects_path = os.path.join(DATA_DIR, "project_info", "project_list.json")
    if os.path.exists(existing_projects_path):
        with open(existing_projects_path, "r", encoding="utf-8") as f:
            existing_projects = json.load(f)
    else:
        existing_projects = []

    # 直接调用
    asyncio.run(analyze_emails(
        df,
        recipient_email,
        kernel=kernel,
        known_projects=existing_projects
    ))
