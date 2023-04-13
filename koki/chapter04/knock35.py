from knock30 import df #形態素解析の結果を格納した辞書, それを格納したデータフレーム

freq = df['surface'].value_counts()#ユニークな要素とその出現回数をシリーズで返す、デフォルトで降順ソート
print(freq)
