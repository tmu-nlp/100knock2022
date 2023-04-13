#knock32
#動詞の基本形をすべて抽出せよ
import knock30

def extract_base(sentences):
    answer = []
    for sentence in sentences:
        for morpheme in sentence:
            if morpheme["pos"] == "動詞":
                print(morpheme["base"])

if __name__ == "__main__":
    fname = "neko.txt.mecab"
    sentences = knock30.load_output(fname)
    extract_base(sentences)