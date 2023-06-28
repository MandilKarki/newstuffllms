import re
import pandas as pd

# Disclaimers and notes
remove_text_en = r"""If you received this email in error, please advise the sender \(by return email or otherwise\) immediately. You have consented to receive the attached electronically at the above-noted email address; please retain a copy of this confirmation for future reference.\s*"""
remove_text_fr = r"""Si vous avez reçu ce courriel par erreur, veuillez en aviser l’expéditeur immédiatement \(par retour de courriel ou un autre moyen\). Vous avez consenti à recevoir le document ci-joint par voie électronique à l’adresse courriel susmentionnée. Veuillez conserver une copie de cette confirmation à titre de référence future.\s*"""
note_text_en = "Note: To protect against computer viruses"
note_text_fr = "Note : Pour se protéger contre les virus informatiques"

def clean_email(raw_email):
    # Remove HTML
    cleaned_email = re.sub(r'<[^>]*?>', '', raw_email)

    # Remove disclaimers and notes
    cleaned_email = re.sub(remove_text_en, '', cleaned_email, flags=re.DOTALL)
    cleaned_email = re.sub(remove_text_fr, '', cleaned_email, flags=re.DOTALL)
    cleaned_email = cleaned_email.replace(note_text_en, '')
    cleaned_email = cleaned_email.replace(note_text_fr, '')

    # Remove URLs
    cleaned_email = re.sub(r'http\S+|www.\S+', '', cleaned_email, flags=re.MULTILINE)
    
    # Remove email addresses
    cleaned_email = re.sub(r'\S*@\S*\s?', '', cleaned_email, flags=re.MULTILINE)
    
    # Remove specific bracketed text
    cleaned_email = re.sub(r'\[cid:image.*?\]', '', cleaned_email, flags=re.MULTILINE)
    
    # Remove lines with "_______________________________________________________________________"
    cleaned_email = re.sub(r'_{10,}', '', cleaned_email, flags=re.MULTILINE)
    
    # Remove line starting with "This email was sent by"
    cleaned_email = re.sub(r'This email was sent by:.*', '', cleaned_email, flags=re.MULTILINE)
    
    # Remove line starting with "To view the list of companies"
    cleaned_email = re.sub(r'To view the list of companies.*', '', cleaned_email, flags=re.MULTILINE)
    
    # Remove line starting with "To unsubscribe, click here:"
    cleaned_email = re.sub(r'To unsubscribe, click here:.*', '', cleaned_email, flags=re.MULTILINE)
    
    # Remove line starting with "Ce courriel a"
    cleaned_email = re.sub(r'Ce courriel a.*', '', cleaned_email, flags=re.MULTILINE)
    
    # Remove line starting with "Pour afficher la liste des sociétés"
    cleaned_email = re.sub(r'Pour afficher la liste des sociétés.*', '', cleaned_email, flags=re.MULTILINE)
    
    # Remove line starting with "Pour vous désabonner, cliquez ici:"
    cleaned_email = re.sub(r'Pour vous désabonner, cliquez ici:.*', '', cleaned_email, flags=re.MULTILINE)
    
    return cleaned_email.strip()

# Assume 'df' is your DataFrame and it has a column 'body' with the raw emails.
df['cleaned_body'] = df['body'].apply(clean_email)
df['cleaned_body_length'] = df['cleaned_body'].apply(len)
df['body_length'] = df['body'].apply(len)

# Display the DataFrame with original and cleaned email bodies, and their lengths
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
    print(df[['body', 'body_length', 'cleaned_body', 'cleaned_body_length']])
