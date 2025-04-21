from fastapi import FastAPI, Request
from msal import ConfidentialClientApplication
import os
import uvicorn
import requests
from dotenv import load_dotenv
import re
from html import unescape

load_dotenv()

app = FastAPI()

# Config from .env or environment variables
CLIENT_ID = os.getenv("APPLICATION_ID")
print("CLIENT_ID: ", CLIENT_ID)
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
AUTHORITY = "https://login.microsoftonline.com/common"
SCOPE = ["Mail.Read", "User.Read"]

msal_app = ConfidentialClientApplication(
    client_id=CLIENT_ID,
    client_credential=CLIENT_SECRET,
    authority=AUTHORITY
)

@app.get("/")
def login():
    auth_url = msal_app.get_authorization_request_url(SCOPE, redirect_uri=REDIRECT_URI)
    return {"auth_url": auth_url}

@app.get("/callback")
def callback(request: Request):
    # Handle OAuth2 callback and exchange code for tokens
    code = request.query_params.get("code")
    result = msal_app.acquire_token_by_authorization_code(
        code,
        scopes=SCOPE,
        redirect_uri=REDIRECT_URI
    )
    access_token = result.get("access_token")
    if not access_token:
        return {"error": result.get("error_description")}

    # Fetch sent emails, selecting only subject and body
    headers = {"Authorization": f"Bearer {access_token}"}
    # Use $select to limit fields
    sent_items_url = (
        "https://graph.microsoft.com/v1.0/me/mailFolders/sentItems/messages"
        "?$top=10&$select=subject,body"
    )
    response = requests.get(sent_items_url, headers=headers)
    if not response.ok:
        return {"error": response.text}

    items = response.json().get("value", [])
    # Map to only subject and HTML body content
    result_list = []
    for msg in items:
        # Extract body content and strip HTML tags to get plain text
        body_content = msg.get("body", {}).get("content", "")
        # Unescape HTML entities
        text = unescape(body_content)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        result_list.append({
            "subject": msg.get("subject", ""),
            "body": text
        })
    print(result_list)
    return result_list

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)