import pandas as pd
df = pd.read_table('popular-names.txt', delimiter='\t', header = None)
df.to_csv('replaced-py.txt', sep = ' ', header=False, index=False)

#pandasを使わない場合
"""
with open('popular-names.txt','r') as f:
    for line in f:
        print(line.strip().replace("\t", " "))
"""