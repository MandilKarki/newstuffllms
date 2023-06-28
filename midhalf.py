# import pandas as pd
# from fuzzywuzzy import fuzz

# def process_emails(filename):
#     df = pd.read_csv(filename)

#     df['self_sent'] = df.apply(lambda row: fuzz.ratio(row['sender'], row['receivers']) > 80, axis=1)

#     df['num_receivers'] = df['receivers'].apply(lambda x: len(x.split(';')))

#     df['indicator'] = df.apply(lambda row: 'self' if row['self_sent'] and row['num_receivers']==1 else 'multi' if row['num_receivers']>10 else 'normal', axis=1)

#     return df

# df = process_emails('emails.csv')
# print(df)
# import pandas as pd
# from fuzzywuzzy import fuzz

# def process_emails(filename):
#     df = pd.read_csv(filename)

#     df['self_sent'] = df.apply(lambda row: any(fuzz.ratio(row['sender'].split('@')[0], receiver.split('@')[0]) > 80 for receiver in row['receivers'].split(';')), axis=1)

#     df['num_receivers'] = df['receivers'].apply(lambda x: len(x.split(';')))

#     df['indicator'] = df.apply(lambda row: 'self' if row['self_sent'] and row['num_receivers']==1 else 'multi' if row['num_receivers']>10 else 'normal', axis=1)

#     return df

# df = process_emails('emails.csv')
# print(df)

import pandas as pd
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b)).size

def process_emails(filename):
    df = pd.read_csv(filename)

    df['self_sent'] = df.apply(lambda row: any(similar(row['sender'].split('@')[0], receiver.split('@')[0]) > 3 for receiver in row['receivers'].split(';')), axis=1)

    df['num_receivers'] = df['receivers'].apply(lambda x: len(x.split(';')))

    df['indicator'] = df.apply(lambda row: 'self' if row['self_sent'] and row['num_receivers']==1 else 'multi' if row['num_receivers']>10 else 'normal', axis=1)

    return df

df = process_emails('emails.csv')
print(df


# import pandas as pd
# import re

# def clean_email(text):
#     # Define patterns to remove
#     patterns = [
#         r'From: .*',  # Remove 'From:' lines
#         r'To: .*',  # Remove 'To:' lines
#         r'Sent: .*',  # Remove 'Sent:' lines
#         r'Subject: .*',  # Remove 'Subject:' lines
#         r'cc: .*',  # Remove 'cc:' lines
#         r'\[.*\.(jpg|png|gif)\]',  # Remove bracket enclosed image file references
#         r'www\..*\.com',  # Remove URLs
#         r'http.*',  # Remove URLs
#         r'_+',  # Remove lengthy underscore dividers
#         r'This email was sent by.*',  # Remove common disclaimer
#         r'Confidentiality Notice.*',  # Remove common disclaimer
#         r'\w+@\w+\.\w+',  # Remove email addresses
#         r'If you received this email in error.*',  # Remove common disclaimer
#         r'Ce courriel a.*',  # Remove French version of the disclaimer
#     ]

#     cleaned_text = text
#     for pattern in patterns:
#         cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE|re.IGNORECASE|re.DOTALL)
    
#     # Remove multiple empty lines
#     cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
#     return cleaned_text.strip()

# # Example usage with a DataFrame 'df' containing a column 'body'
# df = pd.DataFrame({
#     'body': [
#         '''<Your example email text here>''',
#         # More emails...
#     ]
# })

# df['cleaned_body'] = df['body'].apply(clean_email)

      import pandas as pd
import re

def clean_email_body(body):
    # Replace new line characters and following text until space
    body = re.sub("\n.*? ", " ", body)
    
    # Remove anything within square brackets
    body = re.sub("\[.*?\]", " ", body)
    
    # Remove any long string of underscores "__"
    body = re.sub("_+", " ", body)
    
    # Remove disclaimers and similar, usually these are after a long space, or contains "This email was sent by"
    body = re.sub("\s{3,}.*This email was sent by.*", " ", body, flags=re.DOTALL)
    
    # Remove any remaining URLs
    body = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", body)
    
    # Remove email addresses
    body = re.sub('\S*@\S*\s?', ' ', body)
    
    # Remove remaining non-word characters
    body = re.sub("\W", " ", body)
    
    # Remove extra spaces
    body = re.sub("\s+", " ", body)
    
    return body.strip()

# Assuming you have a DataFrame df with a 'body' column containing the emails
df['body_length'] = df['body'].apply(len)
df['cleaned_body'] = df['body'].apply(clean_email_body)
df['cleaned_body_length'] = df['cleaned_body'].apply(len)

# Now you can print and compare
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1):
    print(df[['body', 'cleaned_body', 'body_length', 'cleaned_body_length']])


