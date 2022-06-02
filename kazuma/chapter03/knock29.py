import requests
from knock27 import knock27
d = knock27()
S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": f"File:{d['国旗画像']}",
    "iiprop": "url"
}
R = S.get(url=URL, params=PARAMS)
print(list(R.json()["query"]["pages"].values())[0]["imageinfo"][0]["url"])
