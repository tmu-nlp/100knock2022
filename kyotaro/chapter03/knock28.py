import re

data = open("27.txt", "r").readlines()

pattern1 = r'\[\[ファイル:(?:[^\|]*?\|[^\|]*?\|)?([^\|]*?)\]\]'
pattern2 = r'\<.*?\>'
pattern3 = r'\[https?\:\/\/(?:[^\s]*?\s)?([^\]]*?)\]'
pattern4 = r'\{\{lang\|(?:[^\|]*?\|)?([^\|]*?)\}\}'
pattern5 = r'\{\{仮リンク\|([^\|]*?)\|(?:[^\|]*?\|)?(?:[^\|]*?)\}\}'
pattern6 = r'\{\{([^\|]*?\|)*?([^\|]*?)\}\}'
pattern7 = r'\[\[(?:[^\|]*?)\|([^\|]*?)?\]\]'


for line in data:
    line = line.strip()
    line = re.sub(pattern1, r'\1', line)
    line = re.sub(pattern2, '', line)
    line = re.sub(pattern3, r'\1', line)
    line = re.sub(pattern4, r'\1', line)
    line = re.sub(pattern5, r'\1', line)
    line = re.sub(pattern6, '', line)
    line = re.sub(pattern7, r'\1', line)
    print(line)