'''
24. ファイル参照の抽出
記事から参照されているメディアファイルをすべて抜き出せ．
'''

import re
f = open('jawiki-uk.txt', 'r', encoding='utf-8')
f_out = open('knock24_output.txt', 'w', encoding='utf-8')
data = f.readlines()
for line in data:
    line = line.strip()
    rs = re.findall(r'\[\[ファイル:(.*?)\]\]$', line)
    if rs:
        refer = rs[0].split('|')[0]
        f_out.write(f'{refer}\n')
f.close()
f_out.close()