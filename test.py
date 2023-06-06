import re

def remove_disclaimer(text):
    # Define a pattern for the disclaimer
    pattern = "(This\s(e-?)?mail(\s(is)?\s(intended)?)?.*(confidential|privileged|property))"
    
    # Substitute the pattern with an empty string
    text = re.sub(pattern, '', text, flags=re.IGNORECASE|re.DOTALL)
            
    return text

df['body'] = df['body'].apply(remove_disclaimer)


import re

def remove_disclaimer(text):
    # Define a pattern for the disclaimer
    pattern = "(This\s(e-?)?mail(\s(is)?\s(intended)?)?.*(confidential|privileged|property|If you received this e-?mail in error))"
    
    # Substitute the pattern with an empty string
    text = re.sub(pattern, '', text, flags=re.IGNORECASE|re.DOTALL)
            
    return text

df['body'] = df['body'].apply(remove_disclaimer)



import re

def remove_disclaimer(text):
    # Define a pattern for the disclaimer
    pattern = "(This\s(e-?)?mail(\s(is)?\s(intended)?)?.*(confidential|privileged|property|If you received this e-?mail in error|the information contained in this e-?mail))"
    
    # Substitute the pattern with an empty string
    text = re.sub(pattern, '', text, flags=re.IGNORECASE|re.DOTALL)
            
    return text

df['body'] = df['body'].apply(remove_disclaimer)



import re

def remove_disclaimer(text):
    # Define a pattern for the disclaimer
    pattern = "(This\s(e-?)?mail(\s(is)?\s(intended)?)?.*(confidential|privileged|property|If you received this e-?mail in error|the information contained in this e-?mail|this email was sent to you))"
    
    # Substitute the pattern with an empty string
    text = re.sub(pattern, '', text, flags=re.IGNORECASE|re.DOTALL)
            
    return text

df['body'] = df['body'].apply(remove_disclaimer)


import re

def remove_disclaimer(text):
    # Define a pattern for the disclaimer
    pattern = "(This\s(e-?)?mail.*?(\.|\n)|This\s(e-?)?mail\s(is)?\s(intended)?.*?(\.|\n)|Confidential.*?(\.|\n)|Privileged.*?(\.|\n)|Property.*?(\.|\n))"
    
    # Substitute the pattern with an empty string
    text = re.sub(pattern, '', text, flags=re.IGNORECASE|re.DOTALL)
    
    # Other disclaimer patterns can be added similarly 
    return text

df['body'] = df['body'].apply(remove_disclaimer)


import re

def remove_disclaimer(text):
    # Define a pattern for the disclaimer
    disclaimer_patterns = [
        "This\s(e-?)?mail.*?(\.|\n)",
        "This\s(e-?)?mail\s(is)?\s(intended)?.*?(\.|\n)",
        "Confidential.*?(\.|\n)",
        "Privileged.*?(\.|\n)",
        "Property.*?(\.|\n)",
        # ...add other patterns here...
    ]
    
    # Compile all patterns into one
    pattern = "(" + "|".join(disclaimer_patterns) + ")"
    
    # Separate the email into a main part and a potential disclaimer part
    main_part, _, potential_disclaimer = text.rpartition("\n\n")
    
    # Check if the potential disclaimer matches our pattern
    if re.match(pattern, potential_disclaimer, flags=re.IGNORECASE|re.DOTALL):
        # If it does, remove it
        text = main_part
    
    return text

df['body'] = df['body'].apply(remove_disclaimer)


import re

def remove_disclaimer(text):
    disclaimer_patterns = [
        "This\s(e-?)?mail.*?(\.|\n)",
        "This\s(e-?)?mail\s(is)?\s(intended)?.*?(\.|\n)",
        "Confidential.*?(\.|\n)",
        "Privileged.*?(\.|\n)",
        "Property.*?(\.|\n)",
        # ...add other patterns here...
    ]
    
    pattern = "(" + "|".join(disclaimer_patterns) + ")"
    
    main_part, _, potential_disclaimer = text.rpartition("\n\n")
    
    if re.match(pattern, potential_disclaimer, flags=re.IGNORECASE|re.DOTALL):
        text = main_part
    
    return text

# Example email with a disclaimer
email = """
Hello,

This is an important message.

Regards,
Sender

----------------
Confidentiality Notice: This email and any files transmitted with it are confidential and intended solely for the use of the individual or entity to whom they are addressed. If you have received this email in error please notify the sender.
"""

# Apply the function
clean_email = remove_disclaimer(email)

print(clean_email)


import re

def remove_disclaimer(text):
    disclaimer_patterns = [
        "This\s(e-?)?mail.*?(\.|\n)",
        "This\s(e-?)?mail\s(is)?\s(intended)?.*?(\.|\n)",
        "Confidential.*?(\.|\n)",
        "Privileged.*?(\.|\n)",
        "Property.*?(\.|\n)",
        # ...add other patterns here...
    ]
    
    pattern = "(" + "|".join(disclaimer_patterns) + ")"
    
    # Separate the email into a main part and a potential disclaimer part
    # Here we split by an empty line, a line of dashes, or a line of equal signs
    parts = re.split("\n\n|^-+$|^=+$", text, flags=re.MULTILINE)
    if len(parts) > 1:
        main_parts, potential_disclaimer = parts[:-1], parts[-1]
        
        # Check if the potential disclaimer matches our pattern
        if re.match(pattern, potential_disclaimer, flags=re.IGNORECASE|re.DOTALL):
            # If it does, remove it
            text = "\n\n".join(main_parts)
    
    return text

# Example email with a disclaimer
email = """
Hello,

This is an important message.

Regards,
Sender

----------------
Confidentiality Notice: This email and any files transmitted with it are confidential and intended solely for the use of the individual or entity to whom they are addressed. If you have received this email in error please notify the sender.
"""

# Apply the function
clean_email = remove_disclaimer(email)

print(clean_email)

import pandas as pd

# Load your csv into a DataFrame
df = pd.read_csv('your_file.csv')

# Apply the function to the 'body' column
df['clean_body'] = df['body'].apply(remove_disclaimer)

# Save the result to a new csv file
df.to_csv('cleaned_emails.csv', index=False)





import pandas as pd
import re

def remove_disclaimer(text):
    disclaimer_patterns = [
        "This\s(e-?)?mail.*?(\.|\n)",
        "This\s(e-?)?mail\s(is)?\s(intended)?.*?(\.|\n)",
        "Confidential.*?(\.|\n)",
        "Privileged.*?(\.|\n)",
        "Property.*?(\.|\n)",
        "(If\syou\sreceived|received\sthis).*?(\.|\n)",  # updated pattern
        # ...add other patterns here...
    ]
    
    pattern = "(" + "|".join(disclaimer_patterns) + ")"
    
    parts = re.split("\n\n|^-+$|^=+$", text, flags=re.MULTILINE)
    if len(parts) > 1:
        main_parts, potential_disclaimer = parts[:-1], parts[-1]
        
        if re.match(pattern, potential_disclaimer, flags=re.IGNORECASE|re.DOTALL):
            text = "\n\n".join(main_parts)
    
    return text

# Load your csv into a DataFrame
df = pd.read_csv('your_file.csv')

# Create a copy of the DataFrame
new_df = df.copy()

# Apply the function to the 'body' column in the new DataFrame
new_df['body'] = new_df['body'].apply(remove_disclaimer)


import pandas as pd
import re

def remove_disclaimer(text):
    disclaimer_patterns = [
        "This\s(e-?)?mail.*?(\.|\n)",
        "This\s(e-?)?mail\s(is)?\s(intended)?.*?(\.|\n)",
        "Confidential.*?(\.|\n)",
        "Privileged.*?(\.|\n)",
        "Property.*?(\.|\n)",
        "(If\syou\sreceived|received\sthis|This e-?mail).*?(\.|\n)",  # broadened pattern
        # ...add other patterns here...
    ]
    
    pattern = "(" + "|".join(disclaimer_patterns) + ")"
    
    parts = re.split("\n\n|^-+$|^=+$", text, flags=re.MULTILINE)
    if len(parts) > 1:
        main_parts, potential_disclaimer = parts[:-1], parts[-1]

        print(f"Before:\n{potential_disclaimer}\n")  # print potential disclaimer before removal
        
        if re.search(pattern, potential_disclaimer, flags=re.IGNORECASE|re.DOTALL):
            text = "\n\n".join(main_parts)

        print(f"After:\n{text}\n")  # print text after removal
    
    return text


def remove_disclaimer(text):
    disclaimer_patterns = [
        "This\s(e-?)?mail.*?(\.|\n)",
        "This\s(e-?)?mail\s(is)?\s(intended)?.*?(\.|\n)",
        "Confidential.*?(\.|\n)",
        "Privileged.*?(\.|\n)",
        "Property.*?(\.|\n)",
        "(If\syou\sreceived|received\sthis|This e-?mail).*?(\.|\n)",  # broadened pattern
        # ...add other patterns here...
    ]
    
    pattern = "(" + "|".join(disclaimer_patterns) + ")"
    
    parts = re.split("\n\n|^-+$|^=+$", text, flags=re.MULTILINE)
    if len(parts) > 1:
        main_parts, potential_disclaimer = parts[:-1], parts[-1]
        
        match = re.search(pattern, potential_disclaimer, flags=re.IGNORECASE|re.DOTALL)
        if match:
            print(f"Matched disclaimer:\n{match.group()}\n")  # print matched disclaimer
            text = "\n\n".join(main_parts)
    
    return text


#############################
fast api
from fastapi import FastAPI
from pydantic import BaseModel
from model import generate_text

app = FastAPI()

class Item(BaseModel):
    text: str

@app.post('/predict')
def predict(item: Item):
    output = generate_text(item.text)
    return {'result': output}


load model

import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16)
generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
