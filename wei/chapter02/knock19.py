from collections import defaultdict
import pandas as pd


df = pd.read_table('popular-names.txt',sep='\t', names =['name', 'gender', 'numbers', 'birth'])
name_freq = defaultdict(lambda :0)
for name in df['name']:
    name_freq[name] += 1

for k, v in sorted(name_freq.items(),reverse=True, key=lambda x: x[1]):
    print(k, v)
