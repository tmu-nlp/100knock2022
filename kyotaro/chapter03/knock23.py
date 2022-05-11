import re

with open("20.txt", "r") as data:
    for line in data:
        line.strip()
        pattern = r'^.*(\=\=.*\=\=.*).*$'
        if re.search(pattern, line):
            match = re.findall('=', line)
            print(line + "level :" + str(int(len(match)/2 - 1)))