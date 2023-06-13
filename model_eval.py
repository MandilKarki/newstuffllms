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
    
    
    
    
    
    def classify_email(email, model, tokenizer, max_length=1024):
    prompt = '''Your long prompt goes here '''
    
    try:
        input_ids = tokenizer.encode(prompt + email, return_tensors="pt")
        
        if len(input_ids[0]) > max_length:
            print(f"Warning: Truncating sequence because it is longer than {max_length} tokens.")
            input_ids = input_ids[:, :max_length]

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
    
    except RuntimeError as e:
        print(f"Error: {e}")
        print(f"Skipping this sequence because it's too long for the model to handle.")
        return "Error: Sequence too long"

    
    
    
    There are several ways you can try to speed up your inference:

    Hardware acceleration: If you have a GPU available, make sure you are using it. Most transformer models are computationally intensive and can benefit greatly from GPU acceleration. PyTorch and the transformers library both support CUDA, which can significantly speed up computations.

    Quantization: This is a method that allows for a trade-off between model size, computational latency, and classification accuracy. It reduces the numerical precision of the model's weights, thereby reducing memory requirements and improving computational speed.

    Use batches: Instead of running inference on each sequence individually, you can group them into batches and run inference on each batch. This can lead to more efficient utilization of your computational resources, especially when using a GPU.

    Reduce sequence lengths: The computational complexity of transformer models is quadratic in the sequence length due to the self-attention mechanism. If possible, reduce the length of your sequences, for example by truncating or summarizing the content.

    Use a smaller model: Transformers library provides several smaller versions of most models, e.g., distilgpt2 is a smaller and faster version of gpt-2.

    Optimize transformer settings: Some transformer settings can be tweaked for speed. For example, you can disable the past state in GPT-2 with use_cache=False.

    Pruning: This is a model optimization technique that involves removing unnecessary parts of the neural network, such as weights, neurons, or entire layers. Pruned models are smaller and faster, while typically maintaining a similar level of performance.

    Use a faster tokenizer: Some tokenizers are faster than others. For example, the "Fast" tokenizers provided by Hugging Face are usually quicker than their standard counterparts.

Remember that some of these methods may affect the accuracy of your results, so it's important to find a balance between speed and accuracy that's suitable for your needs.
User
UISNG ALL THESE TECHINQUES HELP ME OUT
ChatGPT

Let's incorporate some of these techniques in the code:

python

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import re

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def preprocess_email(email, disclaimer_starts):
    email = remove_disclaimer(email, disclaimer_starts)
    email = email.replace("\n", " ")
    return email

def remove_disclaimer(email, disclaimer_starts):
    disclaimer_patterns = [re.escape(start) for start in disclaimer_starts]
    disclaimer_pattern = '|'.join(disclaimer_patterns)
    disclaimer_pattern = r"(.*?)(\n\s*\n|-\s*-|" + disclaimer_pattern + ")"
    match = re.match(disclaimer_pattern, email, flags=re.S | re.I)
    if match:
        return match.group(1)
    else:
        return email

def classify_email(email, model, tokenizer, max_length=1024):
    prompt = '''Your long prompt goes here '''
    
    try:
        input_ids = tokenizer(prompt + email, return_tensors="pt", truncation=True, max_length=max_length)
        sequences = pipeline(
            input_ids=input_ids["input_ids"].to(device),
            max_length=min(len(input_ids["input_ids"][0]) + 40, model.config.max_length),
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = sequences[0]["generated_text"]
        final_result = generated_text.split(":")[-1].strip()
        return final_result

    except RuntimeError as e:
        print(f"Error: {e}")
        print(f"Skipping this sequence because it's too long for the model to handle.")
        return "Error: Sequence too long"

def generate_output(df, model, tokenizer):
    df["classification"] = df["email"].apply(lambda x: classify_email(preprocess_email(x, disclaimer_starts), model, tokenizer))
    return df

def save_results(df, output_file):
    df.to_csv(output_file, index=False)

def main(csv_file, output_file):
    df = load_data(csv_file)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", llm_int8_enable_fp32_cpu_offload=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    df = generate_output(df, model, tokenizer)
    save_results(df, output_file)

if __name__ == "__main__":
    csv_file = "input.csv"  # Replace with your input CSV file
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

    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    model_id = "C:/LLM/quantised_bloom/"  # Replace with your model path

    main(csv_file, output_file)

This code incorporates the following changes:

    Using the faster 'fast' tokenizer.
    Adding truncation to the tokenizer to ensure that long sequences are cut down to a length that the model can handle.
    Running the model on a GPU if one is available.
    Applying the classifier function to all emails at once using Pandas' apply function, which is typically faster than a Python loop.
    Ensuring the length of the generated sequence doesn't exceed the model's maximum
    
    
    
     # Extract the classification from generated text (text after "Classification:")
        classification = generated_text.split("Classification:")[-1].strip()
        return classification
    
    
    
    def main(emails):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", llm_int8_enable_fp32_cpu_offload=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    results = []

    for email in emails:
        processed_email = preprocess_email(email, disclaimer_starts)
        classification = classify_email(processed_email, model, tokenizer)
        results.append(classification)

    return results

if __name__ == "__main__":
    emails = ["email1", "email2", "email3", "email4"]  # Replace with your list of emails

    disclaimer_starts = [
        "This email",
        "This e-mail",
        "This mail is intended",
        "This email is intended",
        "Confidential",
        "Privileged",
        "Property"
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    model_id = "C:/LLM/quantised_bloom/"  # Replace with your model path

    results = main(emails)
    print(results)

