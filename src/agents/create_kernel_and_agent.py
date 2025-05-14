import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions import KernelArguments

load_dotenv()

DEFAULT_AI_MODEL = "gpt-4o-mini"
DEFAULT_PROMPT_SETTINGS = PromptExecutionSettings(
    temperature=0.7,
    top_p=0.9,
    max_tokens=500
)

client = AsyncOpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/"
)

def create_kernel():
    kernel = Kernel()
    return kernel

def add_chat_service(kernel: Kernel, model: str = DEFAULT_AI_MODEL, service_id: str = "default-service"):
    chat_service = OpenAIChatCompletion(
        ai_model_id=model,
        async_client=client,
        service_id=service_id
    )
    kernel.add_service(chat_service)
    return kernel.get_prompt_execution_settings_from_service_id(service_id)

def create_agent(
    kernel: Kernel,
    instructions: str,
    service_id: str,
    agent_name: str,
    plugins: list = None,
    settings: PromptExecutionSettings = DEFAULT_PROMPT_SETTINGS,
):
    
    if plugins:
        for plugin in plugins:
            kernel.add_plugin(plugin["instance"], plugin_name=plugin["name"])

    arguments = KernelArguments(settings=settings)

    agent = ChatCompletionAgent(
        kernel=kernel,
        service_id=service_id,
        name=agent_name,
        instructions=instructions,
        arguments=arguments,
    )

    return agent
