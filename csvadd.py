import pandas as pd

# Load the first CSV
df1 = pd.read_csv('file1.csv')

# Load the second CSV
df2 = pd.read_csv('file2.csv')

# Concatenate the two dataframes
df = pd.concat([df1, df2])

# If you want to save the result to a new CSV file:
df.to_csv('combined.csv', index=False)

This code will combine file1.csv and file2.csv into a new dataframe, df, which will be saved as combined.csv. Note that the rows from the second file are appended below the rows from the first file.

If your CSVs have the same columns and you want to combine them side by side, you can use the pd.merge() function instead of pd.concat().

If your CSV files have different columns and you want to combine them, make sure to specify axis=1 in the pd.concat() function, like so: df = pd.concat([df1, df2], axis=1). This will add the columns from df2 to the right of the columns in df1.

Ensure that the order of rows in both dataframes are logically compatible when using axis=1 as it concatenates dataframes side-by-side based on the index.
