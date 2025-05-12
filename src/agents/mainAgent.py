# agent/email_reply_agent.py

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

from .summary_helper import SummaryDataHelper
 
# Load environment variables
load_dotenv()

# Load summary data from directory
summary_helper = SummaryDataHelper("data/summary_data")

# ----------- Plugin Definition -----------
class EmailStylePlugin:
    @kernel_function(description="Get communication style and tone info.")
    async def get_email_style_summary(self, email: str) -> str:
        summary = summary_helper.get_summary_by_email(email)
        if not summary:
            summary = summary_helper.find_best_match(email) or {"note": "default fallback"}
        return json.dumps(summary, indent=2)


# ----------- Kernel & Agent Setup -----------
AI_MODEL = "gpt-4o-mini"
AI_INSTRUCTIONS = (
    "Generate an email reply for the user; adapt tone and style based on the input context. "
    "If a tool is available to get email communication style, use it for better results."
)

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
kernel.add_plugin(EmailStylePlugin(), plugin_name="emailstyle")

service_id = "reply-mail-agent"
chat_service = OpenAIChatCompletion(
    ai_model_id=AI_MODEL,
    async_client=client,
    service_id=service_id
)
kernel.add_service(chat_service)

settings = kernel.get_prompt_execution_settings_from_service_id(service_id)

agent = ChatCompletionAgent(
    kernel=kernel,
    service_id=service_id,
    name="replyEmailAgent",
    instructions=AI_INSTRUCTIONS,
    arguments=KernelArguments(settings=settings),
)


# ----------- Exported Function -----------
async def generate_email_reply(sender_email: str, message: str) -> str:
    chat_history = ChatHistory()
    chat_history.add_user_message(f"Sender: {sender_email}\nMessage: {message}")
    response = ""
    async for content in agent.invoke_stream(chat_history):
        if hasattr(content, "content") and content.content:
            response += content.content
    return response
