# Rename classification columns
df1 = df1.rename(columns={"classification": "classification_model1"})
df2 = df2.rename(columns={"classification": "classification_model2"})
df3 = df3.rename(columns={"classification": "classification_model3"})

# Merge datasets based on common columns
merged_df = df1.merge(df2, on=list(df1.columns.difference(['classification_model1'])))
merged_df = merged_df.merge(df3, on=list(df1.columns.difference(['classification_model1', 'classification_model2'])))

# Determine the majority classification
merged_df['majority_classification'] = (merged_df[['classification_model1', 'classification_model2', 'classification_model3']] == 'Suspicious').sum(axis=1)
merged_df['majority_classification'] = merged_df['majority_classification'].apply(lambda x: 'Suspicious' if x >= 2 else 'Not Suspicious')

# Save to new csv
merged_df.to_csv('merged_results.csv', index=False)




33333333333333333
import pandas as pd

# Assuming you've already read your dataframes
# df1 = pd.read_csv("path_to_dataset1.csv")
# df2 = pd.read_csv("path_to_dataset2.csv")
# df3 = pd.read_csv("path_to_dataset3.csv")

# Renaming classification columns before merging
df1 = df1.rename(columns={"classification": "classification_model1"})
df2 = df2.rename(columns={"classification": "classification_model2"})
df3 = df3.rename(columns={"classification": "classification_model3"})

# Merge on the ID column, defaulting to 'col_0' here
merged_df = df1.merge(df2, on="col_0", how="outer")
merged_df = merged_df.merge(df3, on="col_0", how="outer")

# Majority classification calculation
merged_df['majority_classification'] = merged_df.apply(lambda row: 'Suspicious' 
                                        if sum([row['classification_model1'] == 'Suspicious', 
                                                row['classification_model2'] == 'Suspicious', 
                                                row['classification_model3'] == 'Suspicious']) >= 2 
                                        else 'Not Suspicious', axis=1)

# Save the resulting dataframe
merged_df.to_csv('merged_results.csv', index=False)



import pandas as pd

# Load your datasets
df1 = pd.read_csv("path_to_dataset1.csv")
df2 = pd.read_csv("path_to_dataset2.csv")
df3 = pd.read_csv("path_to_dataset3.csv")

# Merge based on col_0, keeping all columns
merged_df = df1.merge(df2, on="col_0", how="outer", suffixes=('_model1', '_model2'))
merged_df = merged_df.merge(df3, on="col_0", how="outer")

# If 'classification' column exists in df3, rename it for clarity
if 'classification' in merged_df.columns:
    merged_df.rename(columns={'classification': 'classification_model3'}, inplace=True)

def get_majority_class(row):
    count_suspicious = sum([row.get('classification_model1', "") == "Suspicious", 
                            row.get('classification_model2', "") == "Suspicious", 
                            row.get('classification_model3', "") == "Suspicious"])

    return "Suspicious" if count_suspicious > 1 else "Not Suspicious"

merged_df['majority_classification'] = merged_df.apply(get_majority_class, axis=1)

# Save the merged DataFrame with majority classifications and all columns to a new CSV
merged_df.to_csv("merged_results.csv", index=False)


import pandas as pd

# 1. Load the datasets
df1 = pd.read_csv("path_to_dataset1.csv")
df2 = pd.read_csv("path_to_dataset2.csv")
df3 = pd.read_csv("path_to_dataset3.csv")

# 2. Join the datasets using an 'id' column
dfs = df1[['id', 'classification']].merge(df2[['id', 'classification']], on='id', suffixes=('_model1', '_model2'))
dfs = dfs.merge(df3[['id', 'classification']], on='id')
dfs.rename(columns={'classification': 'classification_model3'}, inplace=True)

# 3. Determine the majority vote
def majority_vote(row):
    classifications = [row['classification_model1'], row['classification_model2'], row['classification_model3']]
    return "Suspicious" if classifications.count("Suspicious") >= 2 else "Not Suspicious"

dfs['majority_classification'] = dfs.apply(majority_vote, axis=1)

# 4. Save the results with the original classifications and the majority vote
dfs.to_csv("result_dataset.csv", index=False)


import re

def extract_classification(text):
    pattern = r'### Response.*?Classification:\s*(\w+)'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

# Test
text = '''
### Some other text
Random content
### Response
Subject: some_subject
Body: some_email
Attachments: some_attachments
Classification: Suspicious
'''

print(extract_classification(text))  # Output: Suspicious
df['classification'] = df['generated_text'].apply(extract_classification)



import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# Loading model weights directory
model_id = "C:/LLM/wizardvicuna"

# Loading model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Initialising model
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # Assuming you're using GPU. If CPU, set to -1
)
print("Pipeline initiated")

def text_splitter(text):
    return text[:1950] if isinstance(text, str) else ' '

def load_data(csv_file):
    return pd.read_csv(csv_file)

def preprocess_email(email):
    return text_splitter(email).replace("\n", " ")

def extract_from_response(response, tag):
    start = response.find(f"{tag}:") + len(tag) + 1
    end = response.find("\n", start)
    return response[start:end].strip()

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

def generate_output(df):
    df["classification"] = df.apply(
        lambda row: classify_email(row["cleaned_body"], row["messageSubject"], row["attachments"]), axis=1)
    return df

def save_results(df, output_file):
    df.to_csv(output_file, index=False)

def main(csv_file, output_file):
    df = load_data(csv_file)  # loading csv files for input batch
    print("Data loaded...")
    df = generate_output(df)  # model prediction
    save_results(df, output_file)  # save results to a excel file

if __name__ == "__main__":
    csv_file = "C:\LLM\August\A16august.csv"
    output_file = "resultsAugust16.csv"
    main(csv_file, output_file)
