import re
import spacy

nlp = spacy.load('en_core_web_sm')
for src, dst in [
    ('./Downloads/kftt-data-1.0/data/orig/kyoto-train.en', 'train.spacy.en'),
    ('./Downloads/kftt-data-1.0/data/orig/kyoto-dev.en', 'dev.spacy.en'),
    ('./Downloads/kftt-data-1.0/data/orig/kyoto-test.en', 'test.spacy.en'),
]:
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = nlp.make_doc(x)
            x = ' '.join([doc.text for doc in x])
            print(x, file=g)

# cat ./Downloads/kftt-data-1.0/data/orig/kyoto-train.ja | sed 's/\s+/ /g' | ginzame > train.ginza.ja 
# cat ./Downloads/kftt-data-1.0/data/orig/kyoto-dev.ja | sed 's/\s+/ /g' | ginzame > dev.ginza.ja
# cat ./Downloads/kftt-data-1.0/data/orig/kyoto-test.ja | sed 's/\s+/ /g' | ginzame > test.ginza.ja

for src, dst in [
    ('train.ginza.ja', 'train.spacy.ja'),
    ('dev.ginza.ja', 'dev.spacy.ja'),
    ('test.ginza.ja', 'test.spacy.ja'),
]:
    with open(src) as f:
        lst = []
        tmp = []
        for x in f:
            x = x.strip()
            if x == 'EOS':
                lst.append(' '.join(tmp))
                tmp = []
            elif x != '':
                tmp.append(x.split('\t')[0])
    with open(dst, 'w') as f:
        for line in lst:
            print(line, file=f)