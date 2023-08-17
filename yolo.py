import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# Loading model weights directory
model_id = "C:/LLM/wizardvicuna"

# Loading model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Initialising model
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # Assuming you're using GPU. If CPU, set to -1
)
print("Pipeline initiated")

def text_splitter(text):
    return text[:1950] if isinstance(text, str) else ' '

def load_data(csv_file):
    return pd.read_csv(csv_file)

def preprocess_email(email):
    return text_splitter(email).replace("\n", " ")

def extract_from_response(response, tag):
    start = response.find(f"{tag}:") + len(tag) + 1
    end = response.find("\n", start)
    return response[start:end].strip()

def classify_email(email, subject, attachments):
    # In-context Learning for the given domain
    context = """
    # Context: Data exfiltration, cybersecurity, banking & financial industry, and user behavior analytics.

    Example 1:
    Subject: Wire Transfer Details
    Body: Attached are the details for the wire transfers for the last three months. Please confirm if they are ready for external sharing.
    Attachments: ['Q1_Transfers.xlsx']
    Classification: Suspicious

    Example 2:
    Subject: Vulnerability Report
    Body: Our IT team has discovered a potential vulnerability in our transaction system. Please find the details attached.
    Attachments: ['Vulnerability_Details.docx']
    Classification: Not Suspicious (But important and requires immediate action)

    Example 3:
    Subject: Unusual Login Attempt
    Body: Our system detected an unusual login attempt from a foreign IP address. The details are attached.
    Attachments: ['Login_Audit_Report.pdf']
    Classification: Suspicious

    """

    # Step 1: Subject Analysis
    prompt_subject = context + f"""
    Analyze the email subject to determine its nature.
    Subject: {subject}
    """
    print("Analyzing Subject...")
    response_subject = pipeline(
        text_inputs=prompt_subject,
        max_length=len(prompt_subject) + 50,
    )[0]["generated_text"]
    summary_subject = extract_from_response(response_subject, "Summary")
    
    # Step 2: Body Analysis
    prompt_body = context + f"""
    Given the analysis from the subject: {summary_subject},
    Analyze the email body to determine its nature.
    Body: {email}
    """
    print("Analyzing Body...")
    response_body = pipeline(
        text_inputs=prompt_body,
        max_length=len(prompt_body) + 50,
    )[0]["generated_text"]
    summary_body = extract_from_response(response_body, "Summary")
    
    # Step 3: Attachments Analysis
    prompt_attachments = context + f"""
    Based on the analysis from the subject: {summary_subject},
    and from the body: {summary_body},
    Analyze the attachments to determine their nature.
    Attachments: {attachments}
    """
    print("Analyzing Attachments...")
    response_attachments = pipeline(
        text_inputs=prompt_attachments,
        max_length=len(prompt_attachments) + 50,
    )[0]["generated_text"]
    summary_attachments = extract_from_response(response_attachments, "Summary")

    classification = f"Summary from Subject: {summary_subject}\n" + \
                     f"Summary from Body: {summary_body}\n" + \
                     f"Summary from Attachments: {summary_attachments}"

    return classification

def generate_output(df):
    df["classification"] = df.apply(
        lambda row: classify_email(row["cleaned_body"], row["messageSubject"], row["attachments"]), axis=1)
    return df

def save_results(df, output_file):
    df.to_csv(output_file, index=False)

def main(csv_file, output_file):
    df = load_data(csv_file)  # loading csv files for input batch
    print("Data loaded...")
    df = generate_output(df)  # model prediction
    save_results(df, output_file)  # save results to a excel file

if __name__ == "__main__":
    csv_file = "C:\LLM\August\A16august.csv"
    output_file = "resultsAugust16.csv"
    main(csv_file, output_file)
