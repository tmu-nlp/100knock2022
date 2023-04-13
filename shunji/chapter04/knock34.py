from knock30 import sentences

nouns = []
noun = ''
n = 0

for sentence in sentences:
    for d in sentence:
        if d['pos'] == '名詞':
            noun += d['surface']
            n += 1
        else:
            if n > 1:
                nouns.append(noun)
            noun = ''
            n = 0

print('\n'.join(nouns))
