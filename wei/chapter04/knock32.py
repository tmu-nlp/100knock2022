'''
32. 動詞の基本形
動詞の基本形をすべて抽出せよ。
'''

from knock30 import load_result

def verb_base(morphs_result):
    res = set()
    for morphs in morphs_result:
        for morph in morphs:
            if morph['pos'] == '動詞':
                res.add(morph['base'])

    return res

if __name__ == "__main__":
    file_path = '../data/neko.txt.mecab'
    results = load_result(file_path)
    ans = verb_base(results)
    print(len(ans))
    print(list(ans)[:10])
