import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd
import asyncio
from agents.emailStyleExtractor import analyze_emails
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from openai import AsyncOpenAI
import pytest

@pytest.mark.asyncio
async def test_extract_email_analysis_from_df():
    df = pd.DataFrame([
        {
            "date": "2022-01-01",
            "sender": "phillip.allen@enron.com",
            "recipient": "test.user@enron.com",
            "subject": "Project Kickoff",
            "body": "Let's start the Foo project with Alice and Bob next week."
        },
        {
            "date": "2022-01-02",
            "sender": "phillip.allen@enron.com",
            "recipient": "test.user@enron.com",
            "subject": "Project Foo update",
            "body": "Update: Bob will present the new timeline. See attached."
        },
    ])

    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
    assert GITHUB_TOKEN, "Set GITHUB_TOKEN in your environment!"

    client = AsyncOpenAI(
        api_key=GITHUB_TOKEN,
        base_url="https://models.inference.ai.azure.com/"
    )
    kernel = Kernel()
    chat_service = OpenAIChatCompletion(
        ai_model_id="gpt-4o-mini",
        async_client=client,
        service_id="test-service"
    )
    kernel.add_service(chat_service)

    summary = await analyze_emails(df, "phillip.allen@enron.com", batch_size=1)

    # --- 新结构下的测试断言 ---
    assert isinstance(summary, list), "Expected list of style dicts"
    assert len(summary) > 0, "Summary should contain at least one style"

    for style in summary:
        assert isinstance(style, dict), "Each style should be a dict"
        assert "context" in style
        assert "tone" in style
        assert "greeting" in style
        assert "closing" in style
        assert "patterns" in style
        assert "keywords" in style
        assert "signature" in style

# pytest test_emailStyleExtractor.py
# pytest tests/test_emailStyleExtractor.py