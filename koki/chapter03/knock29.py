import requests #HTTPライブラリ
import re
from knock28 import res_dict #マークアップ除去後のテンプレート

filename = res_dict['国旗画像'] #目的のファイル名「'Flag of the United Kingdom.svg'」を取得

url = 'https://www.mediawiki.org/w/api.php' #Media WikiのAPIエンドポイント
#URLクエリパラメータを指定
params ={
    'action' : 'query', #Fetch data from and about MediaWiki.
    'format' : 'json', #Output data in JSON format.
    'titles' : 'File:' + filename,
    'prop' : 'imageinfo', #API:Imageinfoの使用を明示
    'iiprop' : 'url' #Imageinfoのパラメータiiprop(取得するファイル情報)、ファイルと説明ページへのURLを提供
}

response = requests.get(url=url, params=params) #json形式でresponseの受け取り
result = re.search(r'"url":"(.*?)"', response.text).group(1) #url情報のみ抽出

#print(response.text)
print(result)
