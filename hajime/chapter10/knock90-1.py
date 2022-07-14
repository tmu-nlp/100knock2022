import re
import spacy


def token_ja(input_file, output_file):
    with open(input_file, "r") as i_file, open(output_file, "w") as o_file:
        sent_list = []
        morph = []
        for line in i_file:
            line = line.strip()
            if line == "EOS":
                sent_list.append(" ".join(morph))
                morph = []
            elif line != "":
                morph.append(line.split("\t")[0])
        for line in sent_list:
            print(line, file=o_file)


token_ja("/content/train.ginza.ja", "/content/train.ja")
token_ja("/content/dev.ginza.ja", "/content/dev.ja")
token_ja("/content/test.ginza.ja", "/content/test.ja")

nlp = spacy.load('en_core_web_sm')


def token_en(input_file, output_file):
    with open(input_file, "r") as i_file, open(output_file, "w") as o_file:
        for line in i_file:
            line = line.strip()
            line = re.sub(r"\s+", " ", line)
            line = nlp.make_doc(line)
            line = " ".join([doc.text for doc in line])
            print(line, file=o_file)


token_en("/content/kftt-data-1.0/data/orig/kyoto-train.en", "/content/train.en")
token_en("/content/kftt-data-1.0/data/orig/kyoto-dev.en", "/content/dev.en")
token_en("/content/kftt-data-1.0/data/orig/kyoto-test.en", "/content/test.en")
