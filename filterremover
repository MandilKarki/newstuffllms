import re

def extract_info(emails):
    pattern = re.compile(r"(^From:.*|^To:.*|^Date:.*|^Subject:.*)", re.MULTILINE)
    emails_info = []

    for email in emails:
        email_info = dict(re.findall(pattern, email))
        
        # Extract the body of the email
        body_pattern = "(\n\n)(?s)(.*$)"
        body = re.search(body_pattern, email)
        if body:
            email_info['Body'] = body.group(2)
        else:
            email_info['Body'] = ''

        # Remove unwanted text from the email body
        unwanted_text1 = "If you received this email in error, please advise the sender (by return email or otherwise) immediately. You have consented to receive the attached electronically at the above-noted email address; please retain a copy of this confirmation for future reference."
        unwanted_text2 = "Si vous recevez ce courriel par erreur, veuillez en aviser l'expÃ©diteur immÃ©diatement, par retour de courriel ou par un autre moyen. Vous avez acceptÃ© de recevoir le(s) document(s) ci-joint(s) par voie Ã©lectronique Ã  l'adresse courriel indiquÃ©e ci-dessus; veuillez conserver une copie de cette confirmation pour les fins de reference future."
        
        email_info['Body'] = email_info['Body'].replace(unwanted_text1, '')
        email_info['Body'] = email_info['Body'].replace(unwanted_text2, '')
        
        emails_info.append(email_info)

    return emails_info

import re
from email import policy
from email.parser import BytesParser

remove_text = r"""If you received this email in error, please advise the sender (by return email or otherwise) immediately. You have consented to receive the attached electronically at the above-noted email address; please retain a copy of this confirmation for future reference.\s*Si vous recevez ce courriel par erreur, veuillez en aviser l'expÃ©diteur immÃ©diatement, par retour de courriel ou par un autre moyen. Vous avez acceptÃ© de recevoir le\(s\) document\(s\) ci-joint\(s\) par voie Ã©lectronique Ã  l'adresse courriel indiquÃ©e ci-dessus; veuillez conserver une copie de cette confirmation pour les fins de reference future."""

def clean_email(raw_email):
    email_without_html = re.sub(r'<[^>]*?>', '', raw_email)
    cleaned_email = re.sub(remove_text, '', email_without_html, flags=re.DOTALL)
    return cleaned_email.strip()

def extract_email_info(raw_email_bytes):
    msg = BytesParser(policy=policy.default).parsebytes(raw_email_bytes)
    email_info = {
        "From": msg['From'],
        "To": msg['To'],
        "Date": msg['Date'],
        "Subject": msg['Subject'],
        "Body": clean_email(msg.get_body(preferencelist=("plain")).get_content())
    }
    return email_info
# import re

# def process_email(email_body):
#     email_parts = re.split('Sent:|From:', email_body)

#     cleaned_parts = []
#     for part in email_parts:
#         lines = part.split('\n')

#         content_lines = []
#         for line in lines:
#             # If the line is blank, skip
#             if line.strip() == "":
#                 continue

#             # Skip over image references, URL and email addresses
#             if ("[cid:" not in line and "@" not in line and 
#                 "http" not in line and not re.search(r'\d{3}-\d{3}-\d{4}', line)):
#                 content_lines.append(line)

#         # Join the content lines back into a single string
#         cleaned_part = "\n".join(content_lines).strip()
#         cleaned_parts.append(cleaned_part)

#     # We want to remove any trailing parts that are just a name (signature)
#     while len(cleaned_parts) > 0 and len(cleaned_parts[-1].split()) <= 2:
#         cleaned_parts.pop()

#     return cleaned_parts

import re

def process_email(email_body):
    email_parts = re.split('Sent:|From:', email_body)

    cleaned_parts = []
    for part in email_parts:
        lines = part.split('\n')

        content_lines = []
        for line in lines:
            # If the line is blank, skip
            if line.strip() == "":
                continue

            # Skip over image references, URLs, email addresses, and unwanted portions
            if ("[cid:" not in line and "@" not in line and 
                "http" not in line and not re.search(r'\d{3}-\d{3}-\d{4}', line) and
                "please advise the sender (by return email or otherwise) immediately" not in line and
                "vous avez consenti à recevoir le document ci-joint par voie électronique" not in line):
                content_lines.append(line)

        # Join the content lines back into a single string
        cleaned_part = "\n".join(content_lines).strip()
        cleaned_parts.append(cleaned_part)

    # We want to remove any trailing parts that are just a name (signature)
    while len(cleaned_parts) > 0 and len(cleaned_parts[-1].split()) <= 2:
        cleaned_parts.pop()

    return cleaned_parts
    
    
    import re

# Define the texts to be removed
remove_texts = [
    r"""From:.*\nSent:.*\nTo:.*\nSubject:.*""",
    r"""If you received this email in error, please advise the sender \(by return email or otherwise\) immediately. You have consented to receive the attached electronically at the above-noted email address; please retain a copy of this confirmation for future reference.\s*Si vous recevez ce courriel par erreur, veuillez en aviser l'expÃ©diteur immÃ©diatement, par retour de courriel ou par un autre moyen. Vous avez acceptÃ© de recevoir le\(s\) document\(s\) ci-joint\(s\) par voie Ã©lectronique Ã  l'adresse courriel indiquÃ©e ci-dessus; veuillez conserver une copie de cette confirmation pour les fins de reference future."""
]

# Function to remove multiple texts
def remove_multiple_texts(string, remove_texts):
    for text in remove_texts:
        string = re.sub(text, '', string, flags=re.DOTALL)
    return string
    
   
   #finallllllllllll
   
   
   
   import pandas as pd
import re

def extract_email_content(email_body):
    # Split the email body into lines
    lines = email_body.split('\n')

    # Remove unwanted sections
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("Sent:") or line.startswith("Subject:") or line.startswith("[External]/[Externe]"):
            continue
        if line.startswith("[cid:") or line.startswith("Inforce Business:") or line.startswith("Illustrations:"):
            continue
        if line.startswith("If you received this email in error") or line.startswith("Si vous avez reçu ce courriel par erreur"):
            break
        filtered_lines.append(line)

    # Join the filtered lines to form the email content
    email_content = '\n'.join(filtered_lines).strip()

    # Remove specific sensitive information
    email_content = re.sub(r'\bRBC\b', '[SENSITIVE INFO]', email_content, flags=re.IGNORECASE)

    return email_content


# Read emails from CSV and extract email content
def process_emails(csv_file):
    df = pd.read_csv(csv_file)
    emails_info = []

    for index, row in df.iterrows():
        email_body = row['Body']
        email_content = extract_email_content(email_body)
        emails_info.append({'Email ID': row['ID'], 'Email Content': email_content})

    return emails_info


# Example usage
csv_file = 'emails.csv'
emails_info = process_emails(csv_file)

# Print the extracted email content
for email in emails_info:
    print(f"Email ID: {email['Email ID']}")
    print(f"Email Content:\n{email['Email Content']}\n")
