import re
import sentencepiece as spm


spm.SentencePieceTrainer.Train('--input=kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=0.9998')

sp = spm.SentencePieceProcessor()
sp.Load('kyoto_ja.model')


with open('en-ja/en-ja.bicleaner05.txt', 'r', encoding='utf-8') as f_in:
    data = [x.split('\t') for x in f_in]
    data = [x for x in data if len(x)==4]
    parallel_data = [[x[3], x[4]] for x in data]


with open('jparacrawl.ja', 'w', encoding='utf-8') as f, open('jparacrawl.en', 'w', encoding='utf-8') as g:
    for en, ja in parallel_data:
        print(ja, file=f)
        print(en, file=g)

with open('jparacrawl.ja', 'r', encoding='utf-8') as f, open('train.jparacrawl.ja', 'w', encoding='utf-8') as g:
    for line in f:
        line = line.strip()
        line = re.sub(r'\s+', ' ', line)
        line = sp.encode_as_pieces(line)
        line = ''.join(line)
        print(line, file=g)

