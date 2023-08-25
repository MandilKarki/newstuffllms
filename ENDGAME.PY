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
    device=0  # Set this to the appropriate device, 0 usually means the first GPU, -1 means CPU
)

def load_data(csv_file):
    """Loads data from a CSV."""
    return pd.read_csv(csv_file)

def generate_prompt(entry):
    """Generates a prompt based on the entry."""
    return f"Specific prompt: {entry}"

def generate_text(entry):
    """Generate text based on the entry."""
    if not isinstance(entry, str):
        return None
    prompt = generate_prompt(entry)
    sequences = pipeline(
        text_inputs=prompt,
        max_length=len(prompt) + 50,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
    )
    generated_text = sequences[0]["generated_text"]
    return generated_text

def process_dataframe(df, columns_to_check):
    """Check specified columns for text and generate text if found."""
    for column in columns_to_check:
        new_column_name = f"{column}_generated"
        df[new_column_name] = df[column].apply(generate_text)
    return df

def save_results(df, output_file):
    """Saves the generated text to a CSV file."""
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    csv_file = "C:/LLM/August/A16august.csv"
    output_file = "resultsAugust16.csv"
    columns_to_check = ['col1', 'col2', 'col3', 'col4', 'col5']  # Replace with your actual column names
    df = load_data(csv_file)
    df = process_dataframe(df, columns_to_check)
    save_results(df, output_file)
