import re
from collections import defaultdict

data = open("20.txt", "r").readlines()
data = "".join(data)
ans = defaultdict(lambda: 0)

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

for k, v in ans.items():
  print(k + ':' + v)