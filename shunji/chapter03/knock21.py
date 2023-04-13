import re

pattern = re.compile(r'\[\[Category:.+') # re.Petternオブジェクト

with open('uk.txt', 'r') as f:
    uk = f.read()

print('\n'.join(pattern.findall(uk))) # 4行目を削除してre.findall('\[\[Category:.+', uk)でもいい