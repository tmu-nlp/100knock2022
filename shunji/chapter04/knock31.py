from knock30 import sentences

verb = []

for sentence in sentences:
    for d in sentence:
        if d['pos'] == '動詞':
            verb.append(d['surface'])

print('\n'.join(verb))
