import re

with open('uk.txt', 'r') as f:
    uk = f.read()

pattern = r'(\={2,})\s*(.+?)\s*(\={2,})'
sections = re.findall(pattern, uk)
for section in sections:
    print(section[1] + ':' + str(len(section[0]) - 1))