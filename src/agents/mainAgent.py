import json
from semantic_kernel.contents import ChatHistory
from agents.plugins.email_style_plugin import EmailStylePlugin
from agents.create_kernel_and_agent import create_kernel, add_chat_service, create_agent

AI_INSTRUCTIONS = (
    "Generate an email reply for the user. Adapt tone and style based on the input context. "
    "Use available tools like emailstyle.get_email_style_summary if needed."
)

# 构建 kernel 和 agent
kernel = create_kernel()
settings = add_chat_service(kernel, service_id="reply-mail-agent")
agent = create_agent(
    kernel=kernel,
    instructions=AI_INSTRUCTIONS,
    service_id="reply-mail-agent",
    agent_name="replyEmailAgent",
    plugins=[{"instance": EmailStylePlugin(), "name": "emailstyle"}],
    settings=settings
)

async def generate_email_reply(sender_email: str, message: str, include_debug=False) -> dict:
    chat_history = ChatHistory()
    chat_history.add_user_message(f"Sender: {sender_email}\nMessage: {message}")
    
    response = ""
    async for content in agent.invoke_stream(chat_history):
        if hasattr(content, "content") and content.content:
            response += content.content

    if not include_debug:
        return response

    plugin = EmailStylePlugin()
    style_data = await plugin.get_email_style_summary(sender_email)

    return {
        "reply": response,
        "style_data": json.loads(style_data)
    }
