import pandas as pd

frequency_appear = []
df = pd.read_table('popular-names.txt',sep = '\t', header=None)
frequency_appear = df[0].value_counts()#ユニーク要素の出現個数を返す

#dfに変換、テキスト出力
output_df = pd.DataFrame(frequency_appear)
output_df.to_csv('output-freq_appear.txt',sep = '\t', header=None)
output_df.head()


