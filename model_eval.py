import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import transformers

# Function to remove disclaimers from email text
def remove_disclaimer(email, disclaimer_starts):
    disclaimer_patterns = [re.escape(start) for start in disclaimer_starts]
    disclaimer_pattern = '|'.join(disclaimer_patterns)
    disclaimer_pattern = r"(.*?)(\n\s*\n|-\s*-|" + disclaimer_pattern + ")"
    match = re.match(disclaimer_pattern, email, flags=re.S | re.I)
    if match:
        return match.group(1)
    else:
        return email

# Load the CSV file
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Preprocess the email data
def preprocess_email(email, disclaimer_starts):
    email = remove_disclaimer(email, disclaimer_starts)
    email = email.replace("\n", " ")
    return email

# Perform email classification
def classify_email(email, model, tokenizer):
    sequences = pipeline(email, max_length=100, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    generated_text = sequences[0]["generated_text"]
    prompt_tokens = tokenizer.encode(email, add_special_tokens=False)
    generated_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
    generated_text = tokenizer.decode(generated_tokens[len(prompt_tokens):])
    return generated_text.strip()

# Generate the final classification output
def generate_output(df, model, tokenizer):
    df["classification"] = ""
    for index, row in df.iterrows():
        email_content = row["email"]
        preprocessed_email = preprocess_email(email_content, disclaimer_starts)
        classification = classify_email(preprocessed_email, model, tokenizer)
        df.at[index, "classification"] = classification
    return df

# Save the results to a CSV file
def save_results(df, output_file):
    df.to_csv(output_file, index=False)

# Main function
def main(csv_file, output_file):
    df = load_data(csv_file)
    model_id = "bigscience/bloom-1b7"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device="cuda")
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
    df = generate_output(df, model, tokenizer)
    save_results(df, output_file)

# Entry point
if __name__ == "__main__":
    csv_file = "input_data.csv"  # Replace with your input CSV file
    output_file = "output_data.csv"  # Replace with your output CSV file

    disclaimer_starts = [
        "This email",
        "This e-mail",
        "This mail is intended",
        "This email is intended",
        "Confidential",
        "Privileged",
        "Property"
    ]

    main(csv_file, output_file)
