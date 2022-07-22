import sentencepiece as spm
import re

spm.SentencePieceTrainer.Train('--input=kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=1.0')

sp = spm.SentencePieceProcessor()

sp.Load('kyoto_ja.model')

in_out_path = [ ('kftt-data-1.0/data/orig/kyoto-train.ja', 'kftt-data-1.0/data/sub/train.sub.ja'), ('kftt-data-1.0/data/orig/kyoto-dev.ja', 'kftt-data-1.0/data/sub/dev.sub.ja'),  ('kftt-data-1.0/data/orig/kyoto-test.ja', 'kftt-data-1.0/data/sub/test.sub.ja')]

for src, dst in in_out_path:
    with open(src, "r") as in_data, open(dst, "w") as out_data:
        for line in in_data:
            line = line.strip()
            line = re.sub(r'\s+', ' ', line)
            line = sp.encode_as_pieces(line)
            line = ' '.join(line)
            out_data.write(f'{line}\n')