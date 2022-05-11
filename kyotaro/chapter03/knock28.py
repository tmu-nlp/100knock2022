import re

data = open("27.txt", "r").readlines()

pattern1 = r'\<(.*?)\>'

for line in data:
    line = line.strip()
    line = re.sub(pattern1, '', line)
    print(line)