'''
34. 名詞の連接
名詞の連接(連続して出現する名詞)を最長一致で抽出せよ。
'''

from knock30 import load_result

def longest_nouns(morphs_result):
    res = set()
    nouns = ''
    cnt = 0
    for morphs in morphs_result:
        for morph in morphs:
            if morph['pos'] == '名詞':
                nouns += morph['surface']    # concatenate all nouns
                cnt += 1   # count the number of noun

            else:
                if cnt >= 2:
                    res.add(nouns)
                nouns = ''
                cnt = 0

    return list(res)

if __name__ == "__main__":
    path = '../data/neko.txt.mecab'
    results = load_result(path)
    ans = longest_nouns(results)
    print(ans[:9])