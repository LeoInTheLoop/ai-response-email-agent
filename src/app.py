from fastapi import FastAPI, Request
from msal import ConfidentialClientApplication
import os
import uvicorn
import requests
from dotenv import load_dotenv
import re
from html import unescape
from googleapiclient.discovery import build
from fastapi.responses import JSONResponse
import base64

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

@app.get("/analyze")
async def analyze(access_token: str):
    

    return {"message": "This is a placeholder for the analyze endpoint."}
@app.get("/returnEmail")
async def returnEmail(access_token: str):
    return {"message": "This is a placeholder for the returnEmail endpoint."}


@app.get("/homepage")
async def homepage():
    card = {
        "cards": [
            {
                "header": {
                    "title": "AI Email Reply Generator",
                    "subtitle": "Generate replies in your style",
                    "imageUrl": "https://example.com/logo.png"
                },
                "sections": [
                    {
                        "widgets": [
                            {
                                "textParagraph": {
                                    "text": "Welcome to the AI Email Reply Generator!"
                                }
                            },
                            {
                                "buttonList": {
                                    "buttons": [
                                        {
                                            "text": "Analyze Emails",
                                            "onClick": {
                                                "openLink": {
                                                    "url": "https://your-backend-url.com/analyze"
                                                }
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return JSONResponse(content=card)

@app.post("/contextual")
async def contextual_card(request: Request):
    email_data = await request.json()
    email_subject = email_data.get("messageMetadata", {}).get("subject", "No Subject")
    card = {
        "cards": [
            {
                "header": {
                    "title": "Email Analysis",
                    "subtitle": f"Subject: {email_subject}"
                },
                "sections": [
                    {
                        "widgets": [
                            {
                                "textParagraph": {
                                    "text": "Analyze this email or generate a reply."
                                }
                            },
                            {
                                "buttonList": {
                                    "buttons": [
                                        {
                                            "text": "Generate Reply",
                                            "onClick": {
                                                "action": {
                                                    "function": "generateReply",
                                                    "parameters": [
                                                        {"key": "emailSubject", "value": email_subject}
                                                    ]
                                                }
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return JSONResponse(content=card)


@app.post("/draftReply")
async def draft_reply(request: Request):
    data = await request.json()
    access_token = data.get("access_token")
    email_id = data.get("email_id")
    reply_content = data.get("reply_content")

    if not access_token or not email_id or not reply_content:
        return {"error": "Missing required parameters"}

    # Build the Gmail API service
    credentials = {"token": access_token}
    service = build("gmail", "v1", credentials=credentials)

    # Fetch the original email to get headers
    original_email = service.users().messages().get(userId="me", id=email_id).execute()
    headers = original_email.get("payload", {}).get("headers", [])
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
    to = next((h["value"] for h in headers if h["name"] == "From"), "")

    # Create the reply message
    reply_message = f"To: {to}\nSubject: Re: {subject}\n\n{reply_content}"

    # Create the draft
    draft_body = {
        "message": {
            "raw": base64.urlsafe_b64encode(reply_message.encode("utf-8")).decode("utf-8")
        }
    }
    draft = service.users().drafts().create(userId="me", body=draft_body).execute()

    return {"message": "Draft created successfully", "draft_id": draft.get("id")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)