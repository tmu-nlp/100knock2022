'''
33. 「AのB」
二つの名詞が「の」で連結されている名詞句を抽出せよ。
'''

import knock30

def extract_noun_phrases(morphs_result):
    res = set()
    for morphs in morphs_result:
        for i in range(1, len(morphs)-1):
            if morphs[i-1]['pos'] == '名詞' and morphs[i]['surface'] == 'の' and morphs[i+1]['pos'] == '名詞':
                noun_phrase = morphs[i-1]['surface'] + morphs[i]['surface'] + morphs[i+1]['surface']
                res.add(noun_phrase)
    return res


if __name__ == "__main__":
    path = '../data/neko.txt.mecab'
    results = knock30.load_result(path)
    ans = list(extract_noun_phrases(results))
    print(ans[:9])


'''['石炭の燃殻', '人の意志', '場の横', 'ものの天下', '当時のまま', '芝生の上', '度の願', '勝手の上', '今の名']'''