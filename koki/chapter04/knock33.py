from knock30 import results #形態素解析の結果を格納した配列、内部は辞書型

noun_phrase = [] #名詞句を格納
for i in range(len(results)):
    if (results[i]['pos'] == '名詞') and (results[i+1]['surface'] == 'の') and (results[i+2]['pos'] == '名詞'):
        noun_phrase.append(results[i]['surface'] + results[i+1]['surface'] + results[i+2]['surface'])
        
print(noun_phrase)
