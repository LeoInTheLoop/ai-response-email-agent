import os 
from typing import Annotated
from openai import AsyncOpenAI

from dotenv import load_dotenv

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistory

from semantic_kernel.agents.open_ai import OpenAIAssistantAgent
from semantic_kernel.contents import AuthorRole, ChatMessageContent
from semantic_kernel.functions import kernel_function

from semantic_kernel.connectors.ai import FunctionChoiceBehavior

from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.functions import KernelArguments, kernel_function

import asyncio 

load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/"
)

kernel = Kernel()
service_id = "github-agent"

chat_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=client,
    service_id=service_id
)
kernel.add_service(chat_service)

agent = ChatCompletionAgent(
    kernel=kernel,
    name="PoetAgent",
    instructions="Write poetic responses about Semantic Kernel.",
    arguments=KernelArguments(
        settings=kernel.get_prompt_execution_settings_from_service_id(service_id)
    )
)

async def main():
    print(">>> Entering main()")  # Debug
    chat_history = ChatHistory()

    user_inputs = [
        "hello word",
    ]

    for user_input in user_inputs:
        print(f">>> User input: {user_input}")  # Debug
        chat_history.add_user_message(user_input)

        agent_name: str | None = None
        full_response = ""
        function_calls = []
        function_results = {}

        try:
            async for content in agent.invoke_stream(chat_history):
                # print(f">>> Received content: {content}")  # Debug

                if not agent_name and hasattr(content, 'name'):
                    agent_name = content.name

                for item in content.items:
                    if isinstance(item, FunctionCallContent):
                        call_info = f"Calling: {item.function_name}({item.arguments})"
                        print(">>> FunctionCallContent:", call_info)  # Debug
                        function_calls.append(call_info)
                    elif isinstance(item, FunctionResultContent):
                        result_info = f"Result: {item.result}"
                        print(">>> FunctionResultContent:", result_info)  # Debug
                        function_calls.append(result_info)
                        function_results[item.function_name] = item.result
                #  if we want know if call some func , change this part 
                if (hasattr(content, 'content') and content.content and content.content.strip() and
                    not any(isinstance(item, (FunctionCallContent, FunctionResultContent))
                            for item in content.items)):
                    full_response += content.content
        except Exception as e:
            print(">>> Exception occurred during streaming:", e)  # Debug

        print(">>> Final response:\n", full_response)

#  asyncio.run
if __name__ == "__main__":
    asyncio.run(main())
