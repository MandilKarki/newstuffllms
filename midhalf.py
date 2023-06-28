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
import pandas as pd
from fuzzywuzzy import fuzz

def process_emails(filename):
    df = pd.read_csv(filename)

    df['self_sent'] = df.apply(lambda row: any(fuzz.ratio(row['sender'].split('@')[0], receiver.split('@')[0]) > 80 for receiver in row['receivers'].split(';')), axis=1)

    df['num_receivers'] = df['receivers'].apply(lambda x: len(x.split(';')))

    df['indicator'] = df.apply(lambda row: 'self' if row['self_sent'] and row['num_receivers']==1 else 'multi' if row['num_receivers']>10 else 'normal', axis=1)

    return df

df = process_emails('emails.csv')
print(df)
