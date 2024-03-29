import pandas as pd

# Sample data
df = pd.read_csv('your_data.csv')

# Assuming each segment is of length 17 (as per your description)
length_of_segment = 17

# Split the dataframe into three segments
df1 = df.iloc[:length_of_segment]
df2 = df.iloc[length_of_segment:2*length_of_segment]
df3 = df.iloc[2*length_of_segment:3*length_of_segment]

# Rename classification columns
df1 = df1.rename(columns={"classification": "classification1"})
df2 = df2.rename(columns={"classification": "classification2"})
df3 = df3.rename(columns={"classification": "classification3"})

# Reset index for merging
df1, df2, df3 = df1.reset_index(drop=True), df2.reset_index(drop=True), df3.reset_index(drop=True)

# Merging dataframes based on index
merged_df = pd.concat([df1, df2['classification2'], df3['classification3']], axis=1)

# Determine the majority classification
def get_majority_class(row):
    count_suspicious = sum([row['classification1'] == 'Suspicious', row['classification2'] == 'Suspicious', row['classification3'] == 'Suspicious'])
    return 'Suspicious' if count_suspicious >= 2 else 'Not Suspicious'

merged_df['majority_classification'] = merged_df.apply(get_majority_class, axis=1)

print(merged_df)


import re

def extract_classification(text):
    pattern = r'### Response.*?Classification:\s*(\w+)'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

def classify_email(email, subject, red_flag, attachments, networkSenderIdentifier):
    # ... [rest of the function]

    # After generating the text:
    print("generatingtext")
    generated_text = sequences[0]["generated_text"]
    print(generated_text)

    # Extracting the classification using your function
    classification = extract_classification(generated_text)

    return classification, generated_text  # Return both values

def generate_output(df):
    # Apply the function and split the results into two new columns
    df["classification"], df["generated_text"] = zip(*df.apply(lambda row: classify_email(row["cleaned_body"], row["messageSubject"], row["self_sent"], row["attachments"], row["networkSenderIdentifier"]), axis=1))
    return df




Hello [Recruiter's Name],

I hope this message finds you well. As my co-op is concluding in a week, I've been reflecting on my journey here and am eager to continue contributing to our company in a full-time capacity. I recently came across the [specific role name] position, and I'm genuinely excited about the opportunities it presents. My experience during the co-op and my passion for [specific aspect of the role or field] make me believe that I could be a strong fit for this role.

I would truly appreciate the chance to discuss this opportunity further and understand how I can align my skills with the team's goals. Please let me know a convenient time for you, and we can schedule a conversation.

Thank you for considering my interest, and I'm looking forward to our discussion.

Warm regards,

[Your Name]
[Your Current Role]














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
