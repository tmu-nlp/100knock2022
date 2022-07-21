import sentencepiece as spm
import re, os

#日本語はsentencepiece
spm.SentencePieceTrainer.Train("--input=./data/kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=0.9995")
sp = spm.SentencePieceProcessor()
sp.Load("kyoto_ja.model")

for src, dst in [
    ("data/kftt-data-1.0/data/orig/kyoto-train.ja", "data/kftt-data-1.0/data/bpe/train.sub.ja"),
    ("data/kftt-data-1.0/data/orig/kyoto-dev.ja", "data/kftt-data-1.0/data/bpe/dev.sub.ja"),
    ("data/kftt-data-1.0/data/orig/kyoto-test.ja", "data/kftt-data-1.0/data/bpe/test.sub.ja"),
]:
    with open(src) as f, open(dst, "w") as g:
        for x in f:
            x = x.strip()
            x = re.sub(r"\s+", " ", x)#全ての空白文字を置換
            x = sp.encode_as_pieces(x)#返り値はリストっぽい
            x = " ".join(x)#空白で結合して文字列に
            print(x, file=g)

#英語はsubword-nmt
#os.system：unixコマンドがpythonで使える
os.system("subword-nmt learn-bpe -s 16000 < data/kftt-data-1.0/data/orig/kyoto-train.en > kyoto_en.codes")
for tar in ["train", "dev", "test"]:
    os.system(f"subword-nmt apply-bpe -c kyoto_en.codes < data/kftt-data-1.0/data/orig/kyoto-{tar}.en > data/kftt-data-1.0/data/bpe/{tar}.sub.en")