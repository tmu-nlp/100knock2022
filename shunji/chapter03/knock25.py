import re

with open('uk.txt', 'r') as f:
    uk = f.read()

pattern = r'^\{\{基礎情報.*?$(.*?)^\}\}'
info = re.findall(pattern, uk, re.MULTILINE + re.DOTALL)

pattern = r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))'
result = dict(re.findall(pattern, info[0], re.MULTILINE + re.DOTALL))

f = open('25.txt', 'w')
for k, v in result.items():
    print(k + ': ' + v)
    f.write(k + ': ' + v + '\n')
f.close()