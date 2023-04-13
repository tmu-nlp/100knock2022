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

ans = set()
for s in sen:
    for i in range(1, len(s) - 1):
        if s[i - 1]['pos'] == '名詞' and s[i]['surface'] == 'の' and s[i + 1]['pos'] == '名詞':
            ans.add(s[i - 1]['surface'] + s[i]['surface'] + s[i + 1]['surface'])

for a in list(ans):
    print(a)
