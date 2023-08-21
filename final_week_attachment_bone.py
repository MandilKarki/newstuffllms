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
