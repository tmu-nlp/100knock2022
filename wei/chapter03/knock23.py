'''23. セクション構造
記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ．'''


import re

f_uk = open('jawiki-uk.txt', 'r', encoding='utf-8')
f_out = open('knock23_output.txt', 'w', encoding='utf-8')
data = f_uk.readlines()
for line in data:
    line = line.strip()
    rs = re.match(r'(\=+)(.*?)(\=+)$', line)
    if rs:
        section_name = rs.group(2).strip() #group(1)イコール列 (2)文字列 (3)イコール列  6行目の()で作ったグループに対応？
        section_level = len(rs.group(1)) - 1
                #https://www.javadrive.jp/python/regex/index4.html
        f_out.write(f'レベル{section_level} {section_name}\n')
f_uk.close()
f_out.close()
