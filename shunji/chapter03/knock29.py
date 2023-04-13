import re
import requests
from knock28 import result_rm

S = requests.Session()
URL = "https://commons.wikimedia.org/w/api.php"
PARAMS = {
    "action": "query",
    "format": "json",
    "titles": "File:" + result_rm['国旗画像'],
    "prop": "imageinfo",
    "iiprop":"url"
}
R = S.get(url=URL, params=PARAMS)
DATA = R.json()
PAGES = DATA['query']['pages']
for k, v in PAGES.items():
    print (v['imageinfo'][0]['url'])