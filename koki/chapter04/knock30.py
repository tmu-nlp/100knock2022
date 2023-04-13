#MeCabの出力フォーマットは次の通り
#「表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音」
#タブ区切りで分割, info = [表層形, [品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音]]となるはず
import pandas as pd

filename = 'neko.txt.mecab'

with open(filename) as f:
    text = f.read().split('\n') #read()メソッド ファイル全体を文字列として取得、splitにより改行区切り
    
sentences = []#1文ごとに格納格納
morphs = []#一時格納
results = []#df作成ように文区切りせずにすべて格納

for i, line in enumerate(text):
    tmp_results = {}
    if (line == 'EOS') or (len(line) == 0):#空白,EOSは無視
        sentences.append(morphs)
        morphs = []

    else:
        info = line.split('\t')#タブ区切りで表層形とその他の2分割 info = [表層形, その他]

        info_1 = info[1].split(',')#その他情報をカンマ区切りで分割 
        #info_1 = [品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音]
        
        tmp_dict = {
            'surface' : info[0],#表層形
            'base' : info_1[6],#基本形
            'pos' : info_1[0],#品詞
            'pos1' : info_1[1]#品詞細分類1
        }
        morphs.append(tmp_dict)  

        #dfにも格納
        if tmp_dict['pos'] != '記号':#記号は排除
            results.append(tmp_dict)

df = pd.DataFrame(results, index=None)

#for txt in sentences[2]:
  #print(txt)
  
'''
確認用

df = df['surface'].value_counts()
print(df.head(10))
'''

