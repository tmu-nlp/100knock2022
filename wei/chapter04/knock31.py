'''
31. 動詞
動詞の表層形をすべて抽出せよ。

'''

from knock30 import load_result

def extract_verbs(morphs_result):
    res = set()
    for morphs in morphs_result:
        for morph in morphs:
            if morph['pos'] == '動詞':
                res.add(morph['surface'])

    return res


if __name__ == '__main__':
    file = '../data/neko.txt.mecab'
    results = load_result(file)

    ans = list(extract_verbs(results))
    print(len(ans))
    print(ans[:10])



