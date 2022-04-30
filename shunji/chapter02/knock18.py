import pandas as pd

lines = []
with open('popular-names.txt', 'r') as rf:
    for line in rf:
        lines.append(line.split('\t'))

df = pd.DataFrame(lines).astype({2: int})
df_s = df.sort_values(2, ascending=False)
print(df_s)
