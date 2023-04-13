'''22. カテゴリ名の抽出
記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．'''


import re
f_uk = open('jawiki-uk.txt', 'r', encoding='utf-8')
f_out = open('knock22_output.txt', 'w', encoding='utf-8')
data = f_uk.readlines()
for line in data:
    line = line.strip()

    rs2 = re.search(r'\[\[Category:(.*?)(\|.*)?\]\]$', line)
    if rs2:
        f_out.write(rs2.group(1) + '\n')
f_uk.close()
f_out.close()
