from fastapi import FastAPI, Request, HTTPException, Path, Cookie, Depends
from fastapi.responses import JSONResponse, RedirectResponse, Response
from urllib.parse import quote
from msal import ConfidentialClientApplication
import os
import pandas as pd
import uvicorn
import requests
from dotenv import load_dotenv
import re
from html import unescape
from uuid import uuid4

from agents.emailStyleExtractor import analyze_emails
from agents.mainAgent import generate_email_reply
# important !   at main folder  >PYTHONPATH=src uvicorn app:app --reload 

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
def get_cleaned_emails(folder: str, num: int, access_token: str, fields=None):
    """Fetch and clean emails from a given folder"""
    msgs = fetch_emails(folder, num, access_token, fields=fields)
    return [{
        **msg,
        "body": clean_email_body(msg.get("body", {}).get("content", ""))
    } for msg in msgs]

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
        # my_email = result.get("id_token_claims", {}).get("upn", "")
        # session_store[session_id + "_email"] = my_email

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
            <h2>menu</h2>
            <ul>
                <li><a href="/email/10">📥 check original mail</a></li>
                
                <li><a href="/send/10">📤 check send mail</a></li>
                <li><a href="/styleExtractor/10">📊 analyze email style</a></li>
            </ul>
        </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")

@app.get("/email/{num}")
async def get_emails(
    num: int = Path(..., gt=0, le=50),
    session_id: str = Cookie(default=None)
):
    """Receive email list from inbox folder"""
    access_token = get_token(session_id)
    if not access_token:
        return RedirectResponse("/")

    try:
        emails = get_cleaned_emails("inbox", num, access_token)
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

        emails = get_cleaned_emails("sentItems", num, access_token,fields="subject,body,toRecipients,createdDateTime")
        return {"emails": emails}
    except requests.HTTPError as e:
        error_detail = e.response.json().get("error", {}).get("message", str(e))
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)

@app.get("/styleExtractor/{emailsNum}")
async def style_extractor(
    emailsNum: int = 10,
    session_id: str = Cookie(default=None)
):
    """Analyze email writing style from sent emails"""
    access_token = get_token(session_id)
    if not access_token:
        return RedirectResponse("/")
    try:
        emails = get_cleaned_emails("sentItems", emailsNum, access_token)

        emailsdf = pd.json_normalize(emails)
        print(emailsdf.head()) 
             
        my_email =emailsdf.iloc[0]["from.emailAddress.address"]
        print("my_email:", my_email)
        emailsdf.drop(columns=['@odata.etag','id','from.emailAddress.name',"from.emailAddress.address"], inplace=True)
        # save emailsdf for debug
        emailsdf.to_csv("emailsdf.csv", index=False, encoding="utf-8")  

        style = await analyze_emails(emailsdf, my_email, batch_size=5)

        return {"style": style}

    except requests.HTTPError as e:
        error_detail = e.response.json().get("error", {}).get("message", str(e))
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)

@app.get("/reply/")  #/reply/{email_id}
# todo
async def reply_email(
    email_id: str,
    session_id: str = Cookie(default=None)
):
    """Receive email list from inbox folder"""
    access_token = get_token(session_id)
    if not access_token:
        return RedirectResponse("/")

    try:
        emails = get_cleaned_emails("inbox", 1, access_token)
        # specific email ？
        

    except requests.HTTPError as e:
        error_detail = e.response.json().get("error", {}).get("message", str(e))
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    
    reply_email = await generate_email_reply(
        sender_email=emails[0]["from"]["emailAddress"]["address"],
        message=emails[0]["body"],
        include_debug=True
    )
    return JSONResponse(content=reply_email)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
