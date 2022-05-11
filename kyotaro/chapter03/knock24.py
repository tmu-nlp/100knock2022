import re

with open("20.txt", "r") as data:
    for line in data:
        line.strip()
        pattern = r'^.*\[\[ファイル:(.*?)(?:\|.*).*$'
        if re.search(pattern, line):
            ans = re.findall(pattern, line, re.MULTILINE)
            ans = "\n".join(ans)
            print(ans)