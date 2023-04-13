import pandas as pd

df = pd.read_table('popular-names.txt',sep='\t', names =['name', 'gender', 'numbers', 'birth'])
df = df.sort_values(by='numbers', axis=0, ascending=False)

df.to_csv('knock18.txt', sep='\t', index=False, header=False)