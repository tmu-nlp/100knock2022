import pandas as pd

df = pd.read_table('popular-names.txt', header=None)
col1 = df.iloc[:,0]
col2 = df.iloc[:,1]

new_df = col1 + '\t' + col2
new_df.to_csv('merged_columns-py.txt', header=None, index=None)
