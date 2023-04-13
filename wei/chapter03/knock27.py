'''
27. 内部リンクの除去
26の処理に加えて，テンプレートの値からMediaWikiの内部リンクマークアップを除去し，テキストに変換せよ
'''

import re

f = open('jawiki-uk.txt', 'r', encoding='utf-8')
f_out = open('knock27_output.txt', 'w', encoding='utf-8')
data = f.readlines()
for line in data:
    line = line.strip()
    rs = re.match(r'\|(.*?)\s\=\s*(.*?)$', line)  # 25 基礎情報の抽出
    if rs:
        field_name = rs.group(2)
        rs_empha_delete = re.sub(r'\'+', '', field_name)  # 26 強調マークアップの除去

        rs2 = re.search(r'(.*)\[\[(.*?)\|(.*?)\]\](.*)', rs_empha_delete)
        if rs2:
            if re.match('\[\[ファイル:', rs2.group()):        # [[ファイル は内部リンクマークアップでないので分けて処理
                f_out.write(f'{rs.group(1)} : {rs_empha_delete}\n')
            else:
                field_str = rs2.group(1) + rs2.group(3) + rs2.group(4)
                field_str = re.sub(r'\[\[|\]\]', '', field_str)
                f_out.write(f'{rs.group(1)} : {field_str}\n')

        else:
            rs_inter_link_delete = re.sub(r'\[\[|\]\]', '', rs_empha_delete)
            f_out.write(f'{rs.group(1)} : {rs_inter_link_delete}\n')

f.close()
f_out.close()