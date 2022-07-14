from collections import defaultdict 

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

ans = defaultdict(int) # defaultdict(lambda:0)としても同じ結果が得られる
for s in sen:
    for morph in s:
        if morph['pos'] != '記号':
            ans[morph['base']] += 1  # 単語数を更新する
ans = sorted(ans.items(), key=lambda x: x[1], reverse=True) # 2番目の要素をkeyとして降順でソートする

for fre in ans[:100]:
    print(fre)
