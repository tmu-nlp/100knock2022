import pandas as pd
import re
 
name = "jawiki-country.json"
data = pd.read_json(name, lines = True)
answ = data[data["title"]=="イギリス"]
an = answ["text"].values

for t in an[0].split("\n"):
    if re.search("^=+.*=+$", t):
        num = t.count("=") / 2 - 1
        print(t.replace("=", ""), int(num))
