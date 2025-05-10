import re
def get_email_summary_text(df):
    entries = []
    for idx, row in df.iterrows():
        msg = row.get('message', '')
        # use re
        from_ = re.search(r'From:\s*(.*)', msg)
        to = re.search(r'To:\s*(.*)', msg)
        cc = re.search(r'X-cc:\s*(.*)', msg)
        bcc = re.search(r'X-bcc:\s*(.*)', msg)
        subject = re.search(r'Subject:\s*(.*)', msg)
        date = re.search(r'Date:\s*(.*)', msg)
        
        # body
        body_start = msg.find('\n\n')
        body = msg[body_start+2:] if body_start != -1 else ''
        from_ = from_.group(1).strip() if from_ else ''
        to = to.group(1).strip() if to else ''
        cc = cc.group(1).strip() if cc else ''
        bcc = bcc.group(1).strip() if bcc else ''
        date = date.group(1).strip() if date else ''
        subject = subject.group(1).strip() if subject else ''
        entries.append(
            f"""From: {from_}
To: {to}
CC: {cc}
BCC: {bcc}
Date: {date}
Subject: {subject}
Body: {body.strip()}
-------------------------------"""
        )
    return "\n".join(entries)