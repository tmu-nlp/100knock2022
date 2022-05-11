from knock30 import results #形態素解析の結果を格納した配列、内部は辞書型

longest_noun_phrase = []

for idx in range(len(results)):
    tmp_noun = ''
    if (results[idx]['pos'] == '名詞') and (results[idx + 1]['pos'] == '名詞'):#名詞1語のみは省くため、２語の連なりを判定
        tmp_noun += results[idx]['surface'] + results[idx + 1]['surface']

        cnt = 1
        while(results[idx + 1 + cnt]['pos'] == '名詞'): #以後品詞が名詞でなくなるまで名詞句をtmp_nounに一時格納
            tmp_noun += results[idx + 1 + cnt]['surface']
            cnt += 1

        longest_noun_phrase.append(tmp_noun)

    else:
        continue

print(longest_noun_phrase)
