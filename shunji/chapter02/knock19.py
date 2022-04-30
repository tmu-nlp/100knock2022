import pandas as pd

lines = []
with open('popular-names.txt', 'r') as rf:
    for line in rf:
        lines.append(line.split('\t'))

df = pd.DataFrame(lines)
print(df[0].value_counts())