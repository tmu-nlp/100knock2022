import re

with open("20.txt", "r") as data:
    for line in data:
        line = line.strip()
        pattern = r'^(.*\[\[Category:.*\]\]).*$'
        if re.search(pattern, line):
            print(line)