import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# Loading model weights directory
model_id = "C:/LLM/wizardvicuna"

# Loading model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Initializing the model pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)
print("pipeline initiated")


def text_splitter(text):
    return text[:1950] if isinstance(text, str) else ' '


def load_data(csv_file):
    return pd.read_csv(csv_file)


def preprocess_email(email):
    return text_splitter(email).replace("\n", " ")


def generate_response(prompt):
    sequences = pipeline(
        text_inputs=prompt,
        max_length=len(prompt) + 100,  # Increased to accommodate the additional possible outputs.
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    return sequences[0]["generated_text"]


def classify_email(email, subject, attachments, sender):
    # Analyze Subject:
    subject_result = generate_response(f"Given the email subject '{subject}', determine if there are any indicators of sensitive or client-specific data being present.").split()[-1]

    # Analyze Body:
    body_result = generate_response(f"With the understanding that the subject indicates '{subject_result}', review the body '{email}' for signs of data exfiltration or unauthorized handling.").split()[-1]

    # Analyze Attachments:
    attachment_result = generate_response(f"Given the subject's nature as '{subject_result}' and the body's analysis as '{body_result}', inspect the attachments {attachments} for any evidence of client data or malicious intent.").split()[-1]

    # Final Classification:
    final_classification_prompt = f"Consolidating the analysis: subject ({subject_result}), body ({body_result}), and attachments ({attachment_result}), provide a final classification of the email as 'Suspicious' or 'Not Suspicious'."
    final_classification = generate_response(final_classification_prompt).split("classification:")[-1].strip()

    return final_classification


def generate_output(df):
    df["classification"] = df.apply(
        lambda row: classify_email(row["cleaned_body"], row["messageSubject"], row["attachments"], row["networkSenderIdentifier"]),
        axis=1
    )
    return df


def save_results(df, output_file):
    df.to_csv(output_file, index=False)


def main(csv_file, output_file):
    df = load_data(csv_file)
    print("data loaded")
    print(df)
    df = generate_output(df)
    save_results(df, output_file)


if __name__ == "__main__":
    csv_file = "C:\LLM\August\A16august.csv"
    output_file = "resultsAugust16.csv"
    main(csv_file, output_file)






Analyze Subject:
        Prompt: "Given the email subject {subject}, determine if there are any indicators of sensitive or client-specific data being present."
        Possible Outputs: 'Client Data', 'Sensitive', 'Neutral', 'Unsure'

    Analyze Body:
        Use the output from step 1.
        Prompt: "With the understanding that the subject indicates {outputFromStep1}, review the body {email} for signs of data exfiltration or unauthorized handling."
        Possible Outputs: 'Definitely Suspicious', 'Potentially Suspicious', 'Not Suspicious', 'Unsure'

    Analyze Attachments:
        Use outputs from steps 1 & 2.
        Prompt: "Given the subject's nature as {outputFromStep1} and the body's analysis as {outputFromStep2}, inspect the attachments {attachments} for any evidence of client data or malicious intent."
        Possible Outputs: 'High Risk', 'Moderate Risk', 'Low Risk', 'No Risk'

    Final Classification:
        Use outputs from steps 1, 2 & 3.
        Prompt: "Consolidating the analysis: subject ({outputFromStep1}), body ({outputFromStep2}), and attachments ({outputFromStep3}), provide a final classification of the email as 'Suspicious' or 'Not Suspicious'."






Case 1: Email sent to oneself with client data

Prompt: "Analyze an email sent to {selfEmail} containing the subject {subject}, body {email}, and attachments {attachments}. Given that this email is sent to oneself and contains potential client data, determine if there's evidence of insider threats or unauthorized data handling."

Case 2: Email sent to others with client data

Prompt: "Review an email sent to {recipientEmail} from {senderEmail} with the subject {subject}, body {email}, and attachments {attachments}. Given that this email is sent to another individual and carries possible client data, assess if there's a sign of data exfiltration or suspicious behavior."

Case 3: Email sent to oneself without sensitive information

Prompt: "Examine an email sent to {selfEmail} having the subject {subject}, body {email}, and attachments {attachments}. Considering this email is for personal reference and lacks sensitive information, evaluate if there's any potential risk or subtle signs of malicious intent."

Case 4: Email sent to others without client data

Prompt: "Analyze an email directed to {recipientEmail} from {senderEmail} with the subject {subject}, body {email}, and attachments {attachments}. Given that this email is sent externally but doesn't seem to contain sensitive client data, determine if there are any hidden indications of insider threats."






import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# Loading model weights directory
model_id = "C:/LLM/wizardvicuna"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Initializing the model pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

def load_data(csv_file):
    """Loads data from a CSV."""
    return pd.read_csv(csv_file)

def generate_prompt_based_on_column(column_name, entry):
    """Generates a specific prompt based on the column name."""
    prompts = {
        'column1': f"Specific prompt for {column_name}: {entry}",
        'column2': f"Another prompt for {column_name}: {entry}",
        # ... add prompts for other columns
    }
    return prompts.get(column_name, "")

def classify_entry(column_name, entry):
    """Classify based on the generated prompt."""
    prompt = generate_prompt_based_on_column(column_name, entry)
    sequences = pipeline(
        text_inputs=prompt,
        max_length=len(prompt) + 50,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = sequences[0]["generated_text"]
    classification = generated_text.split("Classification:")[-1].strip()
    return classification

def process_dataframe(df, columns_to_check):
    """Check specified columns for text and classify if found."""
    for column in columns_to_check:
        mask = df[column].apply(lambda x: isinstance(x, str))
        df.loc[mask, column] = df.loc[mask, column].apply(lambda x: classify_entry(column, x))
    return df

def save_results(df, output_file):
    """Saves the classification results to a CSV file."""
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    csv_file = "C:\LLM\August\A16august.csv"
    output_file = "resultsAugust16.csv"
    columns_to_check = ['col1', 'col2', 'col3', 'col4', 'col5'] # Change to your actual column names
    df = load_data(csv_file)
    df = process_dataframe(df, columns_to_check)
    save_results(df, output_file)
