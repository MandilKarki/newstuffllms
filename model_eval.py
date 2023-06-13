import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import transformers

model_id = "C:/LLM/quantised_bloom/"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", llm_int8_enable_fp32_cpu_offload=True)

def remove_disclaimer(email, disclaimer_starts):
    disclaimer_patterns = [re.escape(start) for start in disclaimer_starts]
    disclaimer_pattern = '|'.join(disclaimer_patterns)
    disclaimer_pattern = r"(.*?)(\n\s*\n|-\s*-|" + disclaimer_pattern + ")"
    match = re.match(disclaimer_pattern, email, flags=re.S | re.I)
    if match:
        return match.group(1)
    else:
        return email

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def preprocess_email(email, disclaimer_starts):
    email = remove_disclaimer(email, disclaimer_starts)
    email = email.replace("\n", " ")
    return email

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

def classify_email(email, model, tokenizer):
    prompt = ''' Your long prompt goes here '''
    input_ids = tokenizer.encode(prompt + email, return_tensors="pt")
    sequences = pipeline(
        input_ids=input_ids,
        max_length=len(input_ids[0]) + 40,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    generated_text = sequences[0]["generated_text"]
    final_result = generated_text.split(":")[-1].strip()
    
    return final_result

def generate_output(df, model, tokenizer):
    df["classification"] = ""
    for index, row in df.iterrows():
        email_content = row["email"]
        preprocessed_email = preprocess_email(email_content, disclaimer_starts)
        classification = classify_email(preprocessed_email, model, tokenizer)
        df.at[index, "classification"] = classification
    return df

def save_results(df, output_file):
    df.to_csv(output_file, index=False)

def main(csv_file, output_file):
    df = load_data(csv_file)
    df = generate_output(df, model, tokenizer)
    save_results(df, output_file)

if __name__ == "__main__":
    csv_file = "input.csv"  
    output_file = "output_data.csv"  

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
