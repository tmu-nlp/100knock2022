import re

data = open("26.txt", "r").readlines()

pattern1 = r'\[{2}'
pattern2 = r'\]{2}'

for line in data:
    line = line.strip()
    line = re.sub(pattern1, '', line)
    line = re.sub(pattern2, '', line)
    print(line)