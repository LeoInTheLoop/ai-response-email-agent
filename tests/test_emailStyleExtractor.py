# test_email_analysis.py

import pandas as pd
import asyncio
from src.agents.emailStyleExtractor import extract_email_analysis_from_df
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from openai import AsyncOpenAI
import os

import pytest

@pytest.mark.asyncio
async def test_extract_email_analysis_from_df():
    # Create a small test email DataFrame
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
    # Init fake LLM kernel -- you may want to mock this in a real CI
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

    summary = await extract_email_analysis_from_df(
        df, "phillip.allen@enron.com", kernel, batch_size=1
    )
    # --- Test: Ensure results have correct keys and non-empty content ---
    assert "overall" in summary
    assert any("test.user@" in k for k in summary.keys())
    assert isinstance(summary["overall"], dict)
    # Optional: print to debug
    print(summary)

# pytest test_emailStyleExtractor.py
# pytest tests/test_emailStyleExtractor.py