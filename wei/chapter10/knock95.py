# pip install sentencepiece

import sentencepiece as spm
import re
import spacy

'''
日本語だけのサブワード化を実施
英語のサブワード化、そしてデータのバイナリ処理、機械翻訳モデルの学習、適用(推論即ち日英翻訳)、BLUEスコアの計算は
bashファイルで処理
'''

# 学習の実行
spm.SentencePieceTrainer.Train(
    '--input=/content/kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=0.9995')

# 学習済み単語分割モデルを用いて日本語をトークン化
sp = spm.SentencePieceProcessor()
sp.Load('kyoto_ja.model')

for src, dst in [
    ('kftt-data-1.0/data/orig/kyoto-train.ja', 'train.sub.ja'),
    ('kftt-data-1.0/data/orig/kyoto-dev.ja', 'dev.sub.ja'),
    ('kftt-data-1.0/data/orig/kyoto-test.ja', 'test.sub.ja')
]:
    with open(src, 'r') as rf, open(dst, 'w') as wf:
        for x in rf:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = sp.encode_as_pieces(x)
            x = ' '.join(x)
            print(x, file=wf)


#　システムの翻訳結果をサブワード化

nlp = spacy.load('en_core_web_sm')
def spacy_tokenize(src, dst):
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = ' '.join([doc.text for doc in nlp(x)])
            print(x, file=g)

spacy_tokenize('95.out', '95.out.spacy')