from fastapi import FastAPI, Request, HTTPException, Path, Cookie, Depends
from fastapi.responses import JSONResponse, RedirectResponse, Response
from urllib.parse import quote
from msal import ConfidentialClientApplication
import os
import uvicorn
import requests
from dotenv import load_dotenv
import re
from html import unescape
from uuid import uuid4

load_dotenv()

app = FastAPI()

# Config from .env
CLIENT_ID = os.getenv("APPLICATION_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8000/callback")
AUTHORITY = "https://login.microsoftonline.com/common"
SCOPE = ["Mail.Read", "User.Read"]

msal_app = ConfidentialClientApplication(
    client_id=CLIENT_ID,
    client_credential=CLIENT_SECRET,
    authority=AUTHORITY
)

# Simple in-memory session store (for dev only)
session_store = {}

def get_token(session_id: str):
    return session_store.get(session_id)

def clean_email_body(content: str) -> str:
    """Clean HTML email content to plain text"""
    if not content:
        return ""
    text = unescape(content)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_emails(folder: str, num: int, access_token: str, fields=None):
    if fields is None:
        fields = "subject,body,from"

    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(
        f"https://graph.microsoft.com/v1.0/me/mailFolders/{folder}/messages"
        f"?$top={num}&$select={fields}",
        headers=headers
    )
    response.raise_for_status()
    return response.json().get("value", [])

@app.get("/")
async def login():
    """Redirect to Microsoft login"""
    auth_url = msal_app.get_authorization_request_url(
        scopes=SCOPE,
        redirect_uri=REDIRECT_URI
    )
    return RedirectResponse(auth_url)

@app.get("/callback")
async def callback(code: str):
    """OAuth回调处理并设置session cookie"""
    try:
        result = msal_app.acquire_token_by_authorization_code(
            code=code,
            scopes=SCOPE,
            redirect_uri=REDIRECT_URI
        )

        if "access_token" not in result:
            raise HTTPException(status_code=400, detail="获取token失败")

        session_id = str(uuid4())
        session_store[session_id] = result["access_token"]

        response = RedirectResponse(url="/menu")
        response.set_cookie(key="session_id", value=session_id, httponly=True)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/menu")
async def menu(session_id: str = Cookie(default=None)):
    if not session_id or not get_token(session_id):
        return RedirectResponse("/")

    html_content = """
    <html>
        <body>
            <h2>导航页</h2>
            <ul>
                <li><a href="/cleanemail/10">📥 查看收件箱</a></li>
                <li><a href="/send/10">📤 查看已发送邮件</a></li>
            </ul>
        </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")

@app.get("/cleanemail/{num}")
async def get_clean_emails(
    num: int = Path(..., gt=0, le=50),
    session_id: str = Cookie(default=None)
):
    access_token = get_token(session_id)
    if not access_token:
        return RedirectResponse("/")

    try:
        msgs = fetch_emails("inbox", num, access_token)
        emails = [{
            "subject": msg.get("subject", "No subject"),
            "body": clean_email_body(msg.get("body", {}).get("content", "")),
            "from": msg.get("from", {}).get("emailAddress", {}).get("address", "")
        } for msg in msgs]
        return {"emails": emails}
    except requests.HTTPError as e:
        error_detail = e.response.json().get("error", {}).get("message", str(e))
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)

@app.get("/send/{num}")
async def get_sent_emails(
    num: int = Path(..., gt=0, le=50),
    session_id: str = Cookie(default=None)
):
    access_token = get_token(session_id)
    if not access_token:
        return RedirectResponse("/")

    try:
        msgs = fetch_emails("sentItems", num, access_token, fields="subject,body,toRecipients,createdDateTime")
        emails = [{
            "subject": msg.get("subject", "No subject"),
            "body": clean_email_body(msg.get("body", {}).get("content", "")),
            "to": [r["emailAddress"]["address"] for r in msg.get("toRecipients", [])],
            "date": msg.get("createdDateTime", "")
        } for msg in msgs]
        return {"emails": emails}
    except requests.HTTPError as e:
        error_detail = e.response.json().get("error", {}).get("message", str(e))
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)

@app.get("/styleExtractor")
async def style_extractor(access_token: str, emailsNum: int = 10):
    """TODO: Style extractor for analyzing email writing style"""
    return JSONResponse(content={"message": "Style extractor not implemented yet."})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
