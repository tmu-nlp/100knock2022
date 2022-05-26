import re
from collections import defaultdict

data = open("20.txt", "r").readlines()
ans = defaultdict(lambda: 0)

#テンプレートの部分だけを抜き出す。ただtypeが合わなかったため断念
# data = "".join(data)
# pattern = re.compile(r'^\{\{基礎情報.*?$(.*?)^\}\}', re.MULTILINE + re.DOTALL)
# base = pattern.findall(data)

pattern = r'^\|(.+?)\s+\=\s*(.*)'

for line in data:
  line = line.strip()
  replace = re.findall(pattern, line)
  if replace:
    ans[replace[0][0]] = replace[0][1]

for k, v in ans.items():
  print(k + ":" + v)