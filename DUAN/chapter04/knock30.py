sen = []
mors = []
f = open('./100knock2022/DUAN/chapter04/neko.txt.mecab')
for line in f:  
    if line != 'EOS\n': 
        fields = line.split('\t')
        attr =  fields[1].split(',')
        morph = {'surface': fields[0], 'base': attr[6], 'pos': attr[0], 'pos1': attr[1]}
        mors.append(morph)
    else:  
        sen.append(mors)
        mors = []
for morph in sen[0]:
  print(morph)