import os
import json
import pandas as pd
import asyncio
from dotenv import load_dotenv
from semantic_kernel.agents import ChatCompletionAgent

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from config import DATA_DIR, TRAIN_DATA_PATH

from agents.utils.create_kernel_and_agent import (
    create_agent,
    create_kernel,
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

extract_format = {
    "projects": [
        {
            "name": "Milestone name",
            "due": "YYYY-MM-DD or unknown",
            "completed": True,
            "contributors": {
                "Name1": ["email@example.com", "role"],
                "Name2": ["email2@example.com", "role"]
            },
            "notes": "Optional note or summary",
            "project_keywords": ["extract", "project", "emails"]
        }
    ]
}

INSTRUCTIONS_BASE = """
You are a task finish information extractor.

Given several emails, your task is to identify any ongoing or past projects mentioned in them. For each project, summarize the following:

- projects (name, due date, contributors, completion status, notes)
- members (name, role, email)
- keywords people use to refer to this project


PROJECT IDENTIFICATION RULES:
1. First check against known projects (listed below) by matching:
   - Exact or similar project names
   - Project keywords
   - Team member names
   - Milestone references

2. Only create a new project if:
   - The project is clearly distinct from all known projects
   - It has unique objectives

Please note that the following projects are known to you:
{known_projects}

Please prioritize these known projects by matching project names or keywords from the emails. Avoid creating duplicate project entries that represent the same project.

Output the result as a JSON array. Only extract information that can be inferred directly from the email content — do not make up facts.

===
Format:
{format_json}

Output the result as a JSON array. 
if info unknow ,use '' , if key :'', if name, name else delete the key
if role Contributor,use''

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


    
async def batch_with_halving(
    agent: ChatCompletionAgent,  # The agent instance passed in
    batch_df: pd.DataFrame,
    known_projects: list[dict],
    current_batch_size: int,
    min_batch_size: int = 1,
    recursion_depth: int = 0,
    max_recursion: int = 10
) -> str:
    """
    Process batch recursively, halving batch size when encountering size-related errors.
    
    Args:
        agent: Pre-configured agent instance
        batch_df: DataFrame containing email batch to process
        known_projects: List of known projects for reference
        current_batch_size: Current batch size being attempted
        min_batch_size: Minimum batch size to attempt (default 1)
        recursion_depth: Current recursion depth (default 0)
        max_recursion: Maximum allowed recursion depth (default 10)
        
    Returns:
        Combined JSON response from successful processing
    """
    if recursion_depth >= max_recursion:
        print(f"⚠️ Max recursion depth ({max_recursion}) reached with batch size {current_batch_size}")
        return "[]"
    
    if current_batch_size < min_batch_size:
        print(f"⚠️ Reached minimum batch size {min_batch_size}")
        return "[]"
    
    # Prepare the input data
    current_batch = batch_df.head(current_batch_size)
    emails_text = df_to_text(current_batch)
    prompt_template = format_batch(current_batch, known_projects)
    
    # Update agent instructions with current batch
    # agent.instructions=prompt_template
    
    history = ChatHistory()
    history.add_user_message(emails_text)
    
    try:
        # Attempt processing
        raw_response = ""
        async for part in agent.invoke_stream(history):
            if getattr(part, "content", "").strip():
                raw_response += part.content
        
        if not raw_response:
            print(f"Empty response with batch size {current_batch_size}")
            return "[]"
            
        return raw_response
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error with batch size {current_batch_size}: {error_msg}")
        
        # Check if error is size-related
        if "Request body too large" in error_msg or "too long" in error_msg.lower():
            new_batch_size = max(min_batch_size, current_batch_size // 2)
            print(f"Halving batch size from {current_batch_size} to {new_batch_size}")
            
            # Process first half
            first_half = await batch_with_halving(
                agent,
                batch_df.iloc[:current_batch_size//2],
                known_projects,
                new_batch_size,
                min_batch_size,
                recursion_depth + 1,
                max_recursion
            )
            
            # Process second half
            second_half = await batch_with_halving(
                agent,
                batch_df.iloc[current_batch_size//2:current_batch_size],
                known_projects,
                new_batch_size,
                min_batch_size,
                recursion_depth + 1,
                max_recursion
            )
            
            # Combine results
            try:
                # first_json = json.loads(first_half) if first_half else []
                # second_json = json.loads(second_half) if second_half else []
                combined = first_half + second_half
                return  combined
            except json.JSONDecodeError:
                print("Failed to combine JSON responses")
                return "[]"
        
        # For non-size-related errors, return empty array
        print("Non-size-related error encountered")
        return "[]"
    


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
    instructions = INSTRUCTIONS_BASE.format(
        emails_block="{placeholder}",  # Placeholder for batch text
        format_json=json.dumps(extract_format, ensure_ascii=False, indent=2),
        known_projects=format_known_projects(known_projects)
    )
    # Create agent outside the loop (instructions will be updated per batch)
    agent = create_agent(
        kernel=kernel,
        instructions=instructions,  # Will be updated per batch
        service_id="github-agent",
        agent_name="ExtractAgent",
        settings=PROMPT_SETTING
    )

    for idx, batch_df in enumerate(
        processor.dynamic_batch_generator(df, ["subject", "body"], format_row), start=1
    ):
        print(f"\n---- Processing batch {idx} ----")

        # Get the formatted prompt (but don't use it directly)
        # prompt_template = format_batch(batch_df, known_projects)
        
        # Process with halving logic
        raw_response = await batch_with_halving(
            agent=agent,
            batch_df=batch_df,
            known_projects=known_projects,
            current_batch_size=len(batch_df),
            min_batch_size=1
        )

        if not raw_response:
            print(f"Batch {idx} returned no response.")
            continue

        processor.save_raw_output(raw_response, idx)
        if type(raw_response )== str:
            projects = processor.extract_json_from_response(raw_response)
        print(f"Batch {idx}: extracted {len(projects)} project(s)")

        update_extracted_projects(projects, project_dict)

    # Rest of your function remains the same...
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
