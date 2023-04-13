import pandas as pd

df = pd.read_table('popular-names.txt',sep = '\t', header=None)

N = int(input('分割数を入力'))#N=3
step = int(len(df)/N)

for n in range(N):
    pass
    df_dvided = df.iloc[n * step :(n + 1) * step]
    df_dvided.to_csv('divided_colmuns' + str(n+1) + '.txt',sep = '\t', header = None, index = False)
    
