#knock33
#2つの名詞が「の」で連結されている名詞句を抽出せよ．
import knock30

def extract_noun(sentences):
    for sentence in sentences:
        for i in range(1,len(sentence)-1):#
            if sentence[i-1]["pos"] == "名詞" and sentence[i]["surface"] == "の" and sentence[i+1]["pos"] == "名詞":
                print(sentence[i-1]["surface"] + sentence[i]["surface"] + sentence[i+1]["surface"])

if __name__ == "__main__":
    fname = "neko.txt.mecab"
    sentences = knock30.load_output(fname)
    extract_noun(sentences)
