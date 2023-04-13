#knock31
#動詞の表層形をすべて抽出せよ．
import knock30

def extract_surfce(sentences):
    for sentence in sentences:
        for morpheme in sentence:
            if morpheme["pos"] == "動詞":
                print(morpheme["surface"])

if __name__ == "__main__":
    fname = "neko.txt.mecab"
    sentences = knock30.load_output(fname)
    extract_surfce(sentences)