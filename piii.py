Script 1: Domain Classification

python

def test_classify_domain():
    assert classify_domain("john.doe@gmail.com") == "personal"
    assert classify_domain("jane@abc.com") == "business"
    assert classify_domain("random@outlook.com") == "personal"


Script 2: Extracting unique job titles & checking a column

python

def test_title_exists():
    sample_titles = ["Engineer", "Manager"]
    global job_titles  # Mock the global list
    job_titles = sample_titles
    
    assert title_exists("Engineer") == True
    assert title_exists("Technician") == False
    assert title_exists("Manager") == True


Script 3: Content Labeling

python

def test_classify_content():
    assert classify_content({"body": "Hello, this is not a 404.", "excel": None, "pdf": "data", "powerpoints": None}) == "body with attachments"
    assert classify_content({"body": "Your custom 404 message", "excel": None, "pdf": None, "powerpoints": None}) == "no body no attachments"

4
def test_similarity():
    assert similarity("JohnDoe", ["JohnDoe_resume.pdf", "JD_letter.pdf"]) == 1.0
    assert similarity("John", ["JohnDoe_resume.pdf", "JD_letter.pdf"]) < 1.0
    assert similarity("John", None) == 0

5
def test_contains_PII():
    assert contains_PII("My SSN is 123-45-6789.") == True
    assert contains_PII("My card number is 1234567812345678.") == True
    assert contains_PII("Hello world.") == False







import pandas as pd
import re

# Base patterns for PII checking
patterns = [
    r'\d{3}-\d{2}-\d{4}',  # SSN (SIN) format
    r'\d{16}'  # 16-digit account number
]

# Placeholder for custom words. You can add more as needed.
custom_keywords = [
    "customWord1",
    "customWord2"
]

def detect_PII(text):
    found_pii = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        found_pii.extend(matches)
    
    for keyword in custom_keywords:
        if keyword in text:
            found_pii.append(keyword)

    return found_pii

def process_dataframe(df):
    df['detected_PII'] = df['body'].apply(detect_PII)
    df['contains_PII'] = df['detected_PII'].apply(lambda x: len(x) > 0)
    df['PII_count'] = df['detected_PII'].apply(len)
    return df

