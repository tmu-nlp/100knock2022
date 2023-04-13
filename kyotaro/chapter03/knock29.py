import requests
import re
from collections import defaultdict

data = open("20.txt", "r").readlines()
data = "".join(data)
ans = defaultdict(lambda: 0)

def get_url(text):
    url_file = text['国旗画像'].replace(' ', '_')
    url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + url_file + '&prop=imageinfo&iiprop=url&format=json'
    data = requests.get(url)
    return re.search(r'"url":"(.+?)"', data.text).group(1)

pattern = re.compile(r'^\{\{基礎情報.*?$(.*?)^\}\}', re.MULTILINE + re.DOTALL)
base = pattern.findall(data)
base_a = base[0].split("\n")

for line in base_a:
    key_pattern = r'^.*\|(.*?)(?:\ \=.*).*$'
    key = re.findall(key_pattern, line)
    value_pattern = r'(?:\=)(.*?)$'
    value = re.findall(value_pattern, line)
    value = "".join(value)
    if len(key) >= 1:
        ans[key[0]] = value


print(get_url(ans))