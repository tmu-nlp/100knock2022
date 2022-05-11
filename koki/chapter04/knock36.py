from knock30 import df #形態素解析の結果を格納した辞書, それを格納したデータフレーム
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib #文字化け解消

target = df.index[df['pos'] == '記号']#記号の行を抽出
df = df.drop(target)#記号の行を削除

freq = df['surface'].value_counts()#ユニークな要素とその出現回数をシリーズで返す、デフォルトで降順ソート
freq.head(10).plot(x = 'word', y='frequency', kind = 'bar', figsize = (10,8))#上位10位をmatplotlobで描画、kindでグラフの種類を指定

plt.legend(loc="lower left",bbox_to_anchor=(0.8,1.0))
plt.title('出現頻度上位10語')
plt.xlabel('単語')
plt.ylabel('出現回数')
plt.savefig('output36.png')
plt.show()
