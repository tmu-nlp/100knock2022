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
    nouns = ''
    num = 0
    for morph in s:
        if morph['pos'] == '名詞': 
            nouns = ''.join([nouns, morph['surface']])
            num += 1
        elif num >= 2:  
            ans.add(nouns)
            nouns = ''
            num = 0
        else:  
            nouns = ''
            num = 0
    if num >= 2: 
        ans.add(nouns)

for a in list(ans):
    print(a)
