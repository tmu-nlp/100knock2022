#knock34
#名詞の連接（連続して出現する名詞）を最長一致で抽出せよ
import knock30

def extract_nouns(sentences):
    for sentence in sentences:
        for i in range(0,len(sentence)-1):
            if sentence[i]["pos"] == "名詞" and sentence[i+1]["pos"] == "名詞":
                ans = sentence[i]["surface"]+sentence[i+1]["surface"]
                for j in range(2,len(sentence)-2):
                    if sentence[i+j]["pos"] == "名詞":
                        ans += sentence[i+j]["surface"]
                    else:
                        break
                print(ans)

if __name__ == "__main__":
    fname = "neko.txt.mecab"
    sentences = knock30.load_output(fname)
    extract_nouns(sentences)