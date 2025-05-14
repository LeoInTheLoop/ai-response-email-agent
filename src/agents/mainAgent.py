import json
from semantic_kernel.contents import ChatHistory
from agents.plugins.email_style_plugin import EmailStylePlugin
from agents.utils.create_kernel_and_agent import create_kernel, add_chat_service, create_agent

# chain
# emailanaly agent  ，get email category and return tone from style plugin
#         "context": "describe the typical situation where this style is used",
#         "tone": "summarize the tone, e.g., formal / friendly / direct / humorous",
#         "actor": "who the email is addressed to, e.g., Client / Manager / Colleague / Friend",
#         "intent": "what the email is trying to achieve, e.g., Technical support / Coordination / Request / Follow-up",

# replyanget  use emailanaly already get tone from style plugin 
#  reply


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
