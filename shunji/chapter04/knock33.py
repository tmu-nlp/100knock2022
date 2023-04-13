from knock30 import sentences

noun_phrase = []

for sentence in sentences:
    for i, d in enumerate(sentence):
        if i < 2:
            continue
        else:
            if d['pos'] == '名詞' and sentence[i-1]['surface'] == 'の' and sentence[i-2]['pos'] == '名詞':
                noun_phrase.append(sentence[i-2]['surface'] + sentence[i-1]['surface'] + d['surface'])

print('\n'.join(noun_phrase))
