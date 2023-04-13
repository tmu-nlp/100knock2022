import matplotlib.pyplot as mat
from collections import defaultdict 
import japanize_matplotlib

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

ans = defaultdict(int)
for s in sen:
    for morph in s:
        if morph['pos'] != '記号':
            ans[morph['base']] += 1 
ans = ans.values()

mat.hist(ans, bins=20, range=(1, 20))
mat.xlabel('出現頻度')
mat.ylabel('種類数')
mat.show()
