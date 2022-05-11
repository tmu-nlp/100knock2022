import re

data = open("25.txt", "r").readlines()

pattern = r'\'{2,5}'

for line in data:
    line = line.strip()
    line = re.sub(pattern, '', line)
    print(line)