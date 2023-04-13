'''
26. 強調マークアップの除去
25の処理時に，テンプレートの値からMediaWikiの強調マークアップ（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ
'''

import re
f = open('jawiki-uk.txt', 'r', encoding='utf-8')
f_out = open('knock26_output.txt', 'w', encoding='utf-8')
data = f.readlines()
for line in data:
    line = line.strip()
    rs = re.match(r'\|(.*?)(\s\=\s*)(.*?)$', line)
    if rs:
        field_name = rs.group(3)
        rs_empha_delete = re.sub(r'\'+', '', field_name)
        f_out.write(f'{rs.group(1)} : {rs_empha_delete}\n')

f.close()