import json
from semantic_kernel.functions import kernel_function
from agents.plugins.summary_helper import SummaryDataHelper
from config import DATA_DIR, TRAIN_DATA_PATH
from typing import Annotated

summary_helper = SummaryDataHelper("data/summary_data")

class EmailStylePlugin:
    """Returns email style summary based on role, tone, and intent."""
    @kernel_function(description="Returns a summary of the email style based on input.")
    async def get_email_style_summary(self, input_list: Annotated[list[str], "List of role, tone, and intent."]) -> Annotated[str, "Returns the style description."]:
        role, tone, intent = input_list
        print(f"[Plugin called] role = {role}, tone = {tone}, intent = {intent}")
        query_str = " ".join(input_list)
        summary = summary_helper.find_best_match(query_str)
        if not summary:
            summary = {"note": "No tone match found. Using default style."}
        print("Generated summary:", summary)
        return json.dumps(summary, indent=2)
    



class EmailStylePlugintest:
    def get_email_style_summary(self, input_list):
        # 测试返回内容
        return json.dumps({"test_key": "yougotme"})