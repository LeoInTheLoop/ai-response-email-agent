import json
from semantic_kernel.functions import kernel_function
from .summary_helper import SummaryDataHelper

summary_helper = SummaryDataHelper("data/summary_data")

class EmailStylePlugin:
    @kernel_function(description="Get communication style and tone info.")
    async def get_email_style_summary(self, email: str) -> str:
        summary = summary_helper.get_summary_by_email(email)
        if not summary:
            summary = summary_helper.find_best_match(email) or {"note": "default fallback"}
        return json.dumps(summary, indent=2)
