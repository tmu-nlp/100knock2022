# cabocha -f1 -o ./100knock2022/DUAN/chapter05/ai.ja.txt.parsed ./100knock2022/DUAN/chapter05/ai.ja.txt

import re

class Morph:
    def __init__(self, morph):
        morph = re.split('[\t,]', morph)
        if len(morph) >= 8:
            self.surface = morph[0]
            self.base = morph[7]
            self.pos = morph[1]
            self.pos1 = morph[2]
            
with open('./100knock2022/DUAN/chapter05/ai.ja.txt.parsed') as f_parsed:
    sentences = []
    sentence = []
    for line in f_parsed:
        if line.startswith('*'):
            continue
        elif line.startswith('EOS'):
            if len(sentence) >= 1: 
                sentences.append(sentence)
            sentence = []
        else:
            sentence.append(Morph(line.rstrip()))
        
for morph in sentences[1]:
    print('surface:'+morph.surface, 'base:'+morph.base, 'pos:'+morph.pos, 'pos1:'+morph.pos1)
