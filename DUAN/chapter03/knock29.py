import requests
S = requests.Session()
U = "https://en.wikipedia.org/w/api.php"
 
P = {"action": "query","format": "json","prop": "imageinfo",
          "titles": f"File:{dic['国旗画像']}","iiprop":"url"}
 
R = S.get(url=U, params=P)
data = R.json()
page = data["query"]["pages"]
 
for k, v in page.items():
    print(v["imageinfo"][0]["url"])
