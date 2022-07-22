'''20. JSONデータの読み込み
Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．
問題21-29では，ここで抽出した記事本文に対して実行せよ．'''


import gzip
import json
f = gzip.open('jawiki-country.json.gz')
f_uk = open('jawiki-uk.txt', 'w', encoding='utf-8')
for line in f:
    obj = json.loads(line)     #json.loads()でJSON文字列を辞書に変換
    if obj['title'] == 'イギリス':
        f_uk.write(obj['text'])
f_uk.close()

