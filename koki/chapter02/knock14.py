import pandas as pd

df = pd.read_table('popular-names.txt', header=None)

N = int(input('行数を入力'))#N=3

#下記のうちどれでも良い
print(df[:N])
#print(df.head(N))
#print(df.iloc[:N])  #先頭からN行目まで指定している
