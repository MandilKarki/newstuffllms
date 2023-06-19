import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "C:/LLM/wizardvicuna"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", llm_int8_enable_fp32_cpu_offload=True)

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def preprocess_email(email):
    email = text_splitter(email)
    email = email.replace("\n", " ")
    return email

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

def classify_email(email, model, tokenizer):
    preprocessed_email = preprocess_email(email)

    prompt = ''' Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ...
    
    ## response:

    Email Text:

    '''

    input_ids = prompt + preprocessed_email + " \n Classification:"

    sequences = pipeline(
        text_inputs=input_ids,
        max_length=len(input_ids) + 80,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = sequences[0]["generated_text"]
    classification = generated_text.split("Classification:")[-1].strip()

    return classification

def generate_output(df, model, tokenizer):
    df["classification_body"] = ""

    # Limit to first 10 records for proof of concept
    df = df.head(10)

    for index, row in df.iterrows():
        classification = classify_email(row["body"], model, tokenizer)
        df.at[index, "classification_body"] = classification

    return df

def save_results(df, output_file):
    df.to_csv(output_file, index=False)

def main(csv_file):
    df = load_data(csv_file)
    df = generate_output(df, model, tokenizer)
    save_results(df, csv_file)  # Save results into the same input CSV file

if __name__ == "__main__":
    csv_file = "inputjsoc.csv"  
    main(csv_file)
