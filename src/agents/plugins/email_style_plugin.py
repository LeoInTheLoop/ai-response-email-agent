import json
from semantic_kernel.functions import kernel_function
from agents.summary_helper import SummaryDataHelper

summary_helper = SummaryDataHelper("data/summary_data")

class EmailStylePlugin:
    @kernel_function(description="Get communication style and tone info based on email content.")
    async def get_email_style_summary(self, email_content: str) -> str:
        summary = summary_helper.find_best_match(email_content)
        if not summary:
            summary = {"note": "No tone match found. Using default style."}
        return json.dumps(summary, indent=2)
