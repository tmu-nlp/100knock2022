import re

data = open("26.txt", "r").readlines()

pattern = r'\[\[(?:[^\|]*?\|)??([^\|]*?)\]\]'
#pattern = r'\[\[(?:[^\|]*?)\|([^\|]*?)?\]\]'

for line in data:
    line = line.strip()
    line = re.sub(pattern, r'\1', line)
    print(line)