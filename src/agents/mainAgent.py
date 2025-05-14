# agent/email_reply_agent.py  这代码是为了回复邮件，plugin 是查找适合的语气json文件key是tone ，后续会添加历史查询适合语气， （历史查询project 信息，另一个agent）  ， 怎么改写 ，

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions import KernelArguments
from plugins.email_style_plugin import EmailStylePlugin

load_dotenv()

AI_MODEL = "gpt-4o-mini"
AI_INSTRUCTIONS = (
    "Generate an email reply for the user. Adapt tone and style based on the input context. "
    "Use available tools like emailstyle.get_email_style_summary if needed."
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

def create_kernel_and_agent():
    kernel = Kernel()
    kernel.add_plugin(EmailStylePlugin(), plugin_name="emailstyle")

    chat_service = OpenAIChatCompletion(
        ai_model_id=AI_MODEL,
        async_client=client,
        service_id="reply-mail-agent"
    )
    kernel.add_service(chat_service)

    settings = kernel.get_prompt_execution_settings_from_service_id("reply-mail-agent")

    agent = ChatCompletionAgent(
        kernel=kernel,
        service_id="reply-mail-agent",
        name="replyEmailAgent",
        instructions=AI_INSTRUCTIONS,
        arguments=KernelArguments(settings=settings),
    )

    return kernel, agent

from semantic_kernel.contents import ChatHistory

kernel, agent = create_kernel_and_agent()

async def generate_email_reply(sender_email: str, message: str, include_debug=False) -> dict:
    chat_history = ChatHistory()
    chat_history.add_user_message(f"Sender: {sender_email}\nMessage: {message}")
    
    response = ""
    async for content in agent.invoke_stream(chat_history):
        if hasattr(content, "content") and content.content:
            response += content.content

    if not include_debug:
        return response
    
    # 调用 plugin 直接获取数据
    plugin = EmailStylePlugin()
    style_data = await plugin.get_email_style_summary(sender_email)

    return {
        "reply": response,
        "style_data": json.loads(style_data)
    }
