# new 
# Create a Python function that reads email summary data from summary_data/email_style_projects_summary_total.json. The mainAgent will call this function to retrieve necessary information, then pass the results as input to another agent that generates email replies.
import os
import json
import asyncio
from dotenv import load_dotenv

from openai import AsyncOpenAI

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

load_dotenv()

# ------------------ Your Email Style Data Plugin ------------------
class SummaryDataHelper:
    def __init__(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    def get_summary_for_email(self, email: str):
        return self.data.get(email)
    def get_overall_summary(self):
        return self.data.get('overall', {})

summary_helper = SummaryDataHelper("summary_data/email_style_projects_summary_total.json")

class EmailStylePlugin:
    """
    Plugin that exposes email style summary as a tool.
    """
    @kernel_function(description="Get communication style and tone info for a given email address.")
    async def get_email_style_summary(self, email: str) -> str:
        summary = summary_helper.get_summary_for_email(email)
        if not summary:
            summary = summary_helper.get_overall_summary()
        return json.dumps(summary, indent=2)

# ------------------ Kernel & Agent Setup ------------------
AI_MODEL = "gpt-4o-mini"
AI_INSTRUCTIONS = "Generate an email reply for the user; adapt tone and style based on the input context. If a tool is available to get email communication style, use it for better results."

USER_INPUT_HISTORY = [
    ("john.lavorato@enron.com", "What is the status of the project?"),
    ("david.l.johnson@enron.com", "What is the status of the project?"),
    
]

PROMPT_SETTING = PromptExecutionSettings(
    temperature=0.7,
    top_p=0.9,
    max_tokens=500
)

client = AsyncOpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/"
)

kernel = Kernel()
kernel.add_plugin(EmailStylePlugin(), plugin_name="emailstyle")   # **KEY LINE**

service_id = "reply-mail-agent"
chat_service = OpenAIChatCompletion(
    ai_model_id=AI_MODEL,
    async_client=client,
    service_id=service_id
)
kernel.add_service(chat_service)

settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
# You can also experiment: settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

agent = ChatCompletionAgent(
    kernel=kernel,
    service_id=service_id,
    name="replyEmailAgent",
    instructions=AI_INSTRUCTIONS,
    arguments=KernelArguments(settings=settings),
)

# ------------------ Run Example ------------------
async def main():
    for email, message in USER_INPUT_HISTORY:
        print(f"\n\n== Sender: {email} ==")
        chat_history = ChatHistory()
        chat_history.add_user_message(f"Sender: {email}\nMessage: {message}")
        response = ""
        async for content in agent.invoke_stream(chat_history):
            if hasattr(content, "content") and content.content:
                response += content.content
        print("=== AI Reply ===\n", response)

if __name__ == "__main__":
    asyncio.run(main())