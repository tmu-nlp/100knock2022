'''21. カテゴリ名を含む行を抽出
記事中でカテゴリ名を宣言している行を抽出せよ．
'''

import re
f_uk = open('jawiki-uk.txt', 'r', encoding='utf-8')
f_out = open('knock21_output.txt', 'w', encoding='utf-8')
data = f_uk.readlines()
for line in data:
    rs = re.search(r'\[\[Category:.*\]\]', line)
    if rs:
        f_out.write(rs.group() + '\n')
f_uk.close()
f_out.close()