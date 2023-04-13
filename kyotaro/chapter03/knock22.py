import re

with open("21.txt", "r") as f:
    f = f.readlines()
for line in f:
    line = line.strip()
    pattern = r'^.*\[\[Category:(.*?)(?:\|.*)?\]\].*$'
    print('/n'.join(re.findall(pattern, line, re.MULTILINE)))