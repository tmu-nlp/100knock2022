'''
25. テンプレートの抽出
記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．
'''

import re
f = open('jawiki-uk.txt', 'r', encoding='utf-8')
f_out = open('knock25_output.txt', 'w', encoding='utf-8')
data = f.readlines()
basis_info = dict()
for line in data:
    line = line.strip()
    rs = re.match(r'\|(.*?)(\s\=\s*)(.*?)$', line)
    if rs:
        basis_info[rs.group(1)] = rs.group(3)

for my_key, my_value in basis_info.items():
    f_out.write(f'{my_key} : {my_value}\n')
f.close()
f_out.close()