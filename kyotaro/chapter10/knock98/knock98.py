import tarfile
import sentencepiece as spm
import re

with tarfile.open('en-ja.tar.gz') as tar:
    for f in tar.getmenbers():
        if f.name.endwith('txt'):
            text = tar.extractfile(f).read().decode('utf-8')
            break

data = text.splitlines()
data = [x.split('\t') for x in data]
data = [x for x in data if len(x) == 4]
data = [[x[3], x[2]] for x in data]

with open('jparacrawl.ja', 'w') as f, open('jparacrawl.en', 'w') as g:
    for j, e in data:
        f.write(f'{j}\n')
        g.write(f'{e}\n')

with open('jparacrawl.ja') as f, open('train.jparacrawl.ja', 'w') as g:
    for x in f:
        x = x.strip()
        x = re.sub(r'\s+', ' ', x)
        x = sp.encode_as_pieces(x)
        x = ' '.join(x)
        g.write(f'{x}\n')
        