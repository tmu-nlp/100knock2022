#knock20 JSONデータの読み込み
#Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．
#問題21-29では，ここで抽出した記事本文に対して実行せよ．
import json, gzip

def load_wiki(fname):
    with gzip.open(fname, "r") as f: #ファイルを開く
        for line in f: #一気にloadできないから一行ずつ
            json_data = json.loads(line) #ファイルごとの時はload, 一行ずつの時はloads
            if ("title","イギリス") in json_data.items():
                return json_data["text"]#本文を抽出

if __name__ == "__main__":#あとでload_wiki関数呼び足した時のために
    print(load_wiki("jawiki-country.json.gz"))

