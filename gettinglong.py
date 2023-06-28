import re
import pandas as pd

# Define patterns to remove
remove_patterns = [
    r'<[^>]*?>',  # HTML tags
    r'http\S+|www.\S+',  # URLs
    r'\S*@\S*\s?',  # email addresses
    r'\[cid:image.*?\]',  # specific bracketed text
    r'_{10,}',  # lines with "_______________________________________________________________________"
    r'This email was sent by:.*?$',  # line starting with "This email was sent by"
    r'To view the list of companies.*?$',  # line starting with "To view the list of companies"
    r'To unsubscribe, click here:.*?$',  # line starting with "To unsubscribe, click here:"
    r'Ce courriel a.*?$',  # line starting with "Ce courriel a"
    r'Pour afficher la liste des sociétés.*?$',  # line starting with "Pour afficher la liste des sociétés"
    r'Pour vous désabonner, cliquez ici:.*?$',  # line starting with "Pour vous désabonner, cliquez ici:"
    r'If you received this email in error,.*?(?=\n)',  # English disclaimer and everything after it till end of the line
    r'Si vous avez reçu ce courriel par erreur,.*?(?=\n)',  # French disclaimer and everything after it till end of the line
    r'Note: To protect against computer viruses.*?(?=\n)',  # English note and everything after it till end of the line
    r'Note : Pour se protéger contre les virus informatiques.*?(?=\n)'  # French note and everything after it till end of the line
]

def clean_email(raw_email):
    cleaned_email = raw_email
    for pattern in remove_patterns:
        cleaned_email = re.sub(pattern, '', cleaned_email, flags=re.DOTALL)
    return cleaned_email.strip()

# Assume 'df' is your DataFrame and it has a column 'body' with the raw emails.
df['cleaned_body'] = df['body'].apply(clean_email)
df['cleaned_body_length'] = df['cleaned_body'].apply(len)
df['body_length'] = df['body'].apply(len)

# Display the DataFrame with original and cleaned email bodies, and their lengths
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
    print(df[['body', 'body_length', 'cleaned_body', 'cleaned_body_length']])

# Inspect some of the cleaned emails
for index, row in df.sample(5).iterrows():
    print("Original Email:\n", row['body'], "\nLength:", row['body_length'])
    print("\n")
    print("Cleaned Email:\n", row['cleaned_body'], "\nLength:", row['cleaned_body_length'])
    print("=====================================")
