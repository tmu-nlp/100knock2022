#knock29 国旗画像のURLを取得
import requests
from knock28 import result

flag = result["国旗画像"]
S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": "File:" + flag,
    "iiprop": "url",
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

PAGES = DATA["query"]["pages"]
print(PAGES[list(PAGES)[0]]["imageinfo"][0]["url"])#PAGES[0]だとkey(ページ番号)を指定できないからリストに変換

"""
{'23473560': 
    {'pageid': 23473560, 
     'ns': 6,  
     'title': 'File:Flag of the United Kingdom.svg', 
     'imagerepository': 'local', 
     'imageinfo': 
        [
        {'url': 'https://upload.wikimedia.org/wikipedia/en/a/ae/Flag_of_the_United_Kingdom.svg', 
         'descriptionurl': 'https://en.wikipedia.org/wiki/File:Flag_of_the_United_Kingdom.svg', 
         'descriptionshorturl': 'https://en.wikipedia.org/w/index.php?curid=23473560'
        }
        ]
    }
}
"""