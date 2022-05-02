import re

#前処理
def preproces(txt):
    trans = re.sub(r'[,.]','',text)#re.sub(被置換文字の正規表現, 置換後, 対象)
    l_trans = trans.split(' ')
    return l_trans

text = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'

trg_dict = {}
l_index = (1,5,6,7,8,9,15,16,19)
l_word = preproces(text)

for idx, word in enumerate(l_word):
    if idx + 1 in l_index:
        trg_dict[word[0]] = idx + 1
    else:
        trg_dict[word[0:2]] = idx + 1

print(trg_dict)
