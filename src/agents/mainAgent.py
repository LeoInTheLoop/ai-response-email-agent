import os
os.environ["OTEL_PYTHON_DISABLED"] = "true"
import json
import re
from dotenv import load_dotenv

from typing import Annotated
from openai import AsyncOpenAI

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistory
from semantic_kernel.agents.open_ai import OpenAIAssistantAgent
from semantic_kernel.contents import AuthorRole, ChatMessageContent
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.contents.streaming_text_content import StreamingTextContent

from agents.plugins.email_style_plugin import EmailStylePlugin


# ------------------------------
# Util function to extract <tag>content</tag>
# ------------------------------
def extract_tagged_content(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

# ------------------------------
# Plugin Definition
# ------------------------------
# class EmailStylePlugin:
#     """Returns email style summary based on role, tone, and intent."""

#     @kernel_function(description="Returns a summary of the email style based on input.")
#     def get_email_style_summary(self, input_list: Annotated[list[str], "List of role, tone, and intent."]) -> Annotated[str, "Returns the style description."]:
#         role, tone, intent = input_list
#         print(f"[Plugin called] role = {role}, tone = {tone}, intent = {intent}")
#         return f"Email style summary plug was be use."



# ------------------------------
# Environment and Kernel Setup
# ------------------------------
load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/"
)

kernel = Kernel()
kernel.add_plugin(EmailStylePlugin(), plugin_name="emailstyle")

service_id = "email_reply_agent"
chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o",
    async_client=client,
    service_id=service_id
)
kernel.add_service(chat_completion_service)

settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

AGENT_NAME = "EmailReplyAgent"
AI_INSTRUCTIONS = """
You are a professional email assistant.

Your task is to generate an email reply for the user. To determine the proper tone and writing style,
you must use the plugin: emailstyle.get_email_style_summary. This plugin returns the appropriate
style based on the input.

Use the plugin like this:
> emailstyle.get_email_style_summary(["Client", "Formal", "Request Info"])
always call the plugin to get the style data before generating the reply.

Where:
- role: who the email is for (e.g., Client, Manager, Colleague, Friend, etc.)
- tone: the style and feeling of the message (e.g., Formal, Friendly, Direct)
- intent: what the reply is trying to achieve (e.g., Request Info, Confirm Details)

Output Format (mandatory):
<xml usepluginmethod>
str how agent use the plug like '> emailstyle.get_email_style_summary(["Client", "Formal", "Request Info"])'
</xml>
<xml style_data>
{style JSON returned from plugin} or agent can't use plugin
</xml>
<xml reply>
{your email reply text}
</xml>


"""

agent = ChatCompletionAgent(
    kernel=kernel,
    name=AGENT_NAME,
    instructions=AI_INSTRUCTIONS,
    arguments=KernelArguments(settings=settings)
)

# ------------------------------
# Email Reply Generator
# ------------------------------
async def generate_email_reply(sender_email: str, message: str, include_debug=False) -> dict:
    chat_history = ChatHistory()
    chat_history.add_user_message(f"Sender: {sender_email}\nMessage: {message}")
    print(f"message: {message}")
    response = ""
    function_calls = []
    function_results = []

    # Invoke the agent without any user input to get the introduction

    try:
        async for content in agent.invoke_stream(chat_history):
            if hasattr(content, "content") and content.content:
                response += content.content
            if isinstance(content, StreamingTextContent):
                for item in content.items:
                    if isinstance(item, FunctionCallContent):
                        function_calls.append(f"{item.function_name}{item.arguments}")
                    elif isinstance(item, FunctionResultContent):
                        function_results.append(str(item.result))
    except GeneratorExit:
        print("[Warning] Generator was closed before completion.")
    except Exception as ex:
        print(f"[Error] Exception during streaming: {ex}")

    print(f"original response: {response}")

    if not include_debug:
        return {"raw": response}
    

    try:
        agent_use_plugin = extract_tagged_content(response, "usepluginmethod")
        style_data_str = extract_tagged_content(response, "style_data")
        reply_text = extract_tagged_content(response, "reply")
        style_data = json.loads(style_data_str) if style_data_str else None
    except Exception as ex:
        return {
            "error": f"XML parse error: {ex}",
            "raw_response": response,
            "function_calls": function_calls,
            "function_results": function_results
        }
    

    print(f"reply_text: {reply_text}")
    print(f"style_data: {style_data}")
    print(f"agent_use_plugin: {agent_use_plugin}")
    print(f"[DEBUG] Function calls: {function_calls}")
    print(f"[DEBUG] Function results: {function_results}")

    return {
        "replyText": reply_text,
        "styleAgent": style_data,
        "agent_use_plugin": agent_use_plugin,
        "function_calls": function_calls,
        "function_results": function_results,
        "raw_response": response
    }



