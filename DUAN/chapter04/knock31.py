# 形態素解析結果の読み込み
sen = []
mors = []
f = open('./100knock2022/DUAN/chapter04/neko.txt.mecab')
for line in f:  
    if line != 'EOS\n': 
        fields = line.split('\t')
        if len(fields) != 2 or fields[0] == '':
            continue
        else:
            attr =  fields[1].split(',')
            morph = {'surface': fields[0], 'base': attr[6], 'pos': attr[0], 'pos1': attr[1]}
            mors.append(morph)
    else:  
        sen.append(mors)
        mors = []

ans = set() # セット型を作る
# ループで回して品詞(pos)が動詞であるものの表層形(surface)を格納する
for s in sen: 
    for morph in s:
        if morph['pos'] == '動詞':
            ans.add(morph['surface']) # 重複しない要素のみ保持する

for verb in list(ans):
    print(verb)
