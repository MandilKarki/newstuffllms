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


import pandas as pd
import re

def clean_email(text):
    # Define patterns to remove
    patterns = [
        r'From: .*',  # Remove 'From:' lines
        r'To: .*',  # Remove 'To:' lines
        r'Sent: .*',  # Remove 'Sent:' lines
        r'Subject: .*',  # Remove 'Subject:' lines
        r'cc: .*',  # Remove 'cc:' lines
        r'\[.*\.(jpg|png|gif)\]',  # Remove bracket enclosed image file references
        r'www\..*\.com',  # Remove URLs
        r'http.*',  # Remove URLs
        r'_+',  # Remove lengthy underscore dividers
        r'This email was sent by.*',  # Remove common disclaimer
        r'Confidentiality Notice.*',  # Remove common disclaimer
        r'\w+@\w+\.\w+',  # Remove email addresses
        r'If you received this email in error.*',  # Remove common disclaimer
        r'Ce courriel a.*',  # Remove French version of the disclaimer
    ]

    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE|re.IGNORECASE|re.DOTALL)
    
    # Remove multiple empty lines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

# Example usage with a DataFrame 'df' containing a column 'body'
df = pd.DataFrame({
    'body': [
        '''<Your example email text here>''',
        # More emails...
    ]
})

df['cleaned_body'] = df['body'].apply(clean_email)

