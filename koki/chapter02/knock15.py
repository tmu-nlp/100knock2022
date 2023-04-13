import pandas as pd

df = pd.read_table('popular-names.txt', header=None)

N = int(input('行数を入力'))#N=3
#print(df.tail(N))
#print(df[-N:])
print(df.iloc[-N:])#末尾N行から終端まで指定している
