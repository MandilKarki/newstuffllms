import csv
from transformers import TextGenerationPipeline, GPTNeoForCausalLM, GPT2Tokenizer

# Setting up the model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)  # Use device=0 for GPU

def extract_from_response(response, keyword):
    lines = response.split("\n")
    for line in lines:
        if line.startswith(keyword):
            return line.split(":")[1].strip()
    return ""

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

def main(input_csv, output_csv):
    with open(input_csv, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output_csv, mode='w') as csv_out:
            fieldnames = ["Email", "Subject", "Attachments", "Classification"]
            writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_reader:
                email = row["Email"]
                subject = row["Subject"]
                attachments = row["Attachments"]
                classification = classify_email(email, subject, attachments)
                writer.writerow({"Email": email, "Subject": subject, "Attachments": attachments, "Classification": classification})

if __name__ == "__main__":
    input_file = "path_to_input.csv"  # Replace with your file path
    output_file = "path_to_output.csv"  # Replace with desired output path
    main(input_file, output_file)
