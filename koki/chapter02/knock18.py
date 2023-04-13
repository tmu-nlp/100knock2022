import pandas as pd

df = pd.read_table('popular-names.txt',sep = '\t', header=None)
df = df.sort_values(2, ascending=False)
df.to_csv('sorted-table.txt', sep = '\t', index = False, header=None)
print(df.head(10))
