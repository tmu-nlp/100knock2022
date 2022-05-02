import pandas as pd

df = pd.read_table('popular-names.txt', header=None)
col1 = df.iloc[:,0]
col2 = df.iloc[:,1]

col1.to_csv('col1.txt',index=False, header=False)
col2.to_csv('col2.txt',index=False, header=False)
