import pandas as pd
import re
from email import policy
from email.parser import BytesParser

# Text to remove
remove_text_en = r"""If you received this email in error, please advise the sender \(by return email or otherwise\) immediately. You have consented to receive the attached electronically at the above-noted email address; please retain a copy of this confirmation for future reference.\s*"""
remove_text_fr = r"""Si vous avez reçu ce courriel par erreur, veuillez en aviser l’expéditeur immédiatement \(par retour de courriel ou un autre moyen\). Vous avez consenti à recevoir le document ci-joint par voie électronique à l’adresse courriel susmentionnée. Veuillez conserver une copie de cette confirmation à titre de référence future.\s*"""
note_text_en = "Note: To protect against computer viruses"
note_text_fr = "Note : Pour se protéger contre les virus informatiques"

def clean_email(raw_email):
    email_without_html = re.sub(r'<[^>]*?>', '', raw_email)
    cleaned_email = re.sub(remove_text_en, '', email_without_html, flags=re.DOTALL)
    cleaned_email = re.sub(remove_text_fr, '', cleaned_email, flags=re.DOTALL)
    cleaned_email = cleaned_email.replace(note_text_en, '')
    cleaned_email = cleaned_email.replace(note_text_fr, '')
    return cleaned_email.strip()

def extract_email_info(raw_email_bytes):
    msg = BytesParser(policy=policy.default).parsebytes(raw_email_bytes)
    email_info = {
        "From": msg['From'],
        "To": msg['To'],
        "Date": msg['Date'],
        "Subject": msg['Subject'],
        "Body": msg.get_body(preferencelist=("plain")).get_content()
    }
    return email_info

# Load the raw emails
with open("path_to_your_emails.txt", "rb") as f:
    raw_emails = f.read().split(b"From nobody ")
    
# Remove the first element which is empty
raw_emails = raw_emails[1:]

# Extract the info and clean the body
emails_info = [extract_email_info(raw_email) for raw_email in raw_emails]
for email in emails_info:
    email["Body"] = clean_email(email["Body"])

df = pd.DataFrame(emails_info)

# Print info
for i, row in df.head(5).iterrows():
    print("-"*80)
    print(f"Email {i+1}:")
    print(f"From: {row['From']}")
    print(f"To: {row['To']}")
    print(f"Date: {row['Date']}")
    print(f"Subject: {row['Subject']}")
    print(f"Body length before cleaning: {len(row['Body'])}")
    cleaned_email = clean_email(row['Body'])
    print(f"Body length after cleaning: {len(cleaned_email)}")
    print(f"Body before cleaning:\n{row['Body']}")
    print(f"Body after cleaning:\n{cleaned_email}\n")