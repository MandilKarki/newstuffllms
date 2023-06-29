import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from dolly_model.instruct_pipeline import InstructionTextGenerationPipeline

tokenizer = AutoTokenizer.from_pretrained("C:/LLM/dolly_model", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("C:/LLM/dolly_model", torch_dtype=torch.bfloat16,
                                             trust_remote_code=True, llm_int8_enable_fp32_cpu_offload=True, device_map='auto')

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

# Read the CSV file
df = pd.read_csv('data.csv') # Replace 'data.csv' with your file path

# Create a new column 'classification' and initialize it with None
df['classification'] = None

# Loop over each row in the dataframe
for i, row in df.iterrows():
    # Create the prompt
    prompt = f'''
    ### Instruction: The task is to just output a classification label and stop and not generate any more.
    ... # Add the rest of the instruction here

    Real case:
    Red Flag: {row['red_flag']}
    Email Text: {row['email_body']}
    Subject: {row['subject']}
    Classification:
    '''

    # Generate the response
    res = generate_text(prompt)

    # Assign the classification result to the 'classification' column
    df.at[i, 'classification'] = res[0]["generated_text"].strip()

# Save the dataframe to a new CSV file
df.to_csv('classified_emails.csv', index=False)
