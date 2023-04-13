import re

with open('uk.txt', 'r') as f:
    uk = f.read()

pattern = r'\[\[ファイル:(.*?)(?:\|.*)*\]\]'
print('\n'.join(re.findall(pattern, uk)))