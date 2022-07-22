import sentencepiece as spm
import re
import spacy

spm.SentencePieceTrainer.Train(
    '--input=kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=1.0')
sp = spm.SentencePieceProcessor()
sp.Load('kyoto_ja.model')


def token_ja_bpe(input_file, output_file):
    with open(input_file, "r") as i_file, open(output_file, "w") as o_file:
        for line in i_file:
            line = line.strip()
            line = re.sub(r"\s+", " ", line)
            line = sp.encode_as_pieces(line)
            line = " ".join(line)
            print(line, file=o_file)


token_ja_bpe("kftt-data-1.0/data/orig/kyoto-train.ja", "train.sub.ja")
token_ja_bpe("kftt-data-1.0/data/orig/kyoto-dev.ja", "dev.sub.ja")
token_ja_bpe("kftt-data-1.0/data/orig/kyoto-test.ja", "test.sub.ja")

nlp = spacy.load('en_core_web_sm')


def token_en(input_file, output_file):
    with open(input_file, "r") as i_file, open(output_file, "w") as o_file:
        for line in i_file:
            line = line.strip()
            line = re.sub(r"\s+", " ", line)
            line = nlp.make_doc(line)
            line = " ".join([doc.text for doc in line])
            print(line, file=o_file)


token_en("/content/kftt-data-1.0/data/orig/kyoto-test.en", "/content/test.en")
