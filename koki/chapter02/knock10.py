import pandas as pd

df = pd.read_table('popular-names.txt', header=None) #対象のtxtにはヘッダ情報がないことに注意
print(len(df))

#pandasを使わない場合
with open('popular-names.txt','r') as f:
    count = 0
    for line in f:
        count += 1
    print(count)
