import json
import re
from semantic_kernel.contents import ChatHistory
from agents.plugins.email_style_plugin import EmailStylePlugin
from agents.utils.create_kernel_and_agent import create_kernel, add_chat_service, create_agent
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, StreamingTextContent


AI_INSTRUCTIONS = """
You are a professional email assistant.

Your task is to generate an email reply for the user. To determine the proper tone and writing style,
you must use the plugin: emailstyle.get_email_style_summary. This plugin returns the appropriate
style based on the input.

Use the plugin like this:
> emailstyle.get_email_style_summary(["Client", "Formal", "Request Info"])
always use the plugin to get the style data before generating the reply.

Where:
- role: who the email is for (e.g., Client, Manager, Colleague, Friend, etc.)
- tone: the style and feeling of the message (e.g., Formal, Friendly, Technical)
- intent: what the email is trying to achieve (e.g., Clarify, Request Info, Follow-up)

📥 Input Format:
[role, tone, intent]

📤 Output Format (mandatory):
<xml usepluginmethod>
str how agent use the plug like '> emailstyle.get_email_style_summary(["Client", "Formal", "Request Info"])'
</xml>
<xml style_data>
{style JSON returned from plugin} or agent can't use plugin
</xml>
<xml reply>
{your email reply text}
</xml>

📘 Reference Combinations (role + tone + intent):
1. {"Client": ["Formal", "Technical"]} — Clarify / Request Info  
2. {"Client": ["Formal", "Polite"]} — Follow-up / Update  
3. {"Manager": ["Formal", "Direct"]} — Request Approval  
4. {"Manager": ["Concise", "Data-driven"]} — Provide Updates  
5. {"Colleague": ["Friendly", "Direct"]} — Coordination  
6. {"Colleague": ["Casual", "Collaborative"]} — Brainstorming  
7. {"Investor": ["Formal", "Persuasive"]} — Proposal / Pitch  
8. {"Investor": ["Transparent", "Detailed"]} — Financial Reporting  
9. {"Friend": ["Humorous", "Casual"]} — Personal Request  
10. {"Friend": ["Warm", "Supportive"]} — Check-in / Catch-up  
11. {"Vendor": ["Professional", "Firm"]} — Negotiation / Issue Resolution  
12. {"Vendor": ["Polite", "Urgent"]} — Logistics Follow-up  
13. {"General": ["Neutral", "Polite"]} — General Purpose
"""


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


def extract_tagged_content(text: str, tag: str) -> str:
    match = re.search(f"<xml {tag}>(.*?)</xml>", text, re.DOTALL)
    return match.group(1).strip() if match else None


async def generate_email_reply(sender_email: str, message: str, include_debug=False) -> dict:
    chat_history = ChatHistory()
    chat_history.add_user_message(f"Sender: {sender_email}\nMessage: {message}")
    # print(f"generate_email_replyChat history: {chat_history}")

    response = ""
    function_calls = []
    function_results = {}



    async for content in agent.invoke_stream(chat_history):
        if hasattr(content, "content") and content.content:
            response += content.content
        if isinstance(content, StreamingTextContent):
            # Track function calls and results
            for item in content.items:
                if isinstance(item, FunctionCallContent):
                    call_info = f"Calling: {item.function_name}({item.arguments})"
                    print(call_info)
                    function_calls.append(call_info)
                elif isinstance(item, FunctionResultContent):
                    result_info = f"Result: {item.result}"
                    print(result_info)
                    function_calls.append(result_info)
                    function_results[item.function_name] = item.result




    if not include_debug:
        return response  
    
    

    # 提取 XML 内容
    try:
        style_data_raw = response.split("<xml style_data>")[1].split("</xml>")[0].strip()
        reply_text = response.split("<xml reply>")[1].split("</xml>")[0].strip()
        agent_use_plugin = response.split("<xml usepluginmethod>")[1].split("</xml>")[0].strip()
        style_data = json.loads(style_data_raw)
    except Exception as e:
        print(f"[Error parsing XML from response]: {e}")
        return {
            "error": "Failed to parse response.",
            "raw_response": response
        }
    print(f'reply_text: {reply_text}')
    print(f'style_data: {style_data}')
    print(f'agent_use_plugin: {agent_use_plugin}')

    print(f"[DEBUG] Function calls: {function_calls}")
    print(f"[DEBUG] Function results: {function_results}")

    return {
        "replyText": reply_text,
        "styleAgent": style_data,
        "raw_response": response if include_debug else None
    }
