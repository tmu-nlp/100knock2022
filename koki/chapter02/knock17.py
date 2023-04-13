import pandas as pd

df = pd.read_table('popular-names.txt',sep = '\t', header=None)

uniq_set = df[0].unique()
uniq_set.sort()
print(uniq_set)
