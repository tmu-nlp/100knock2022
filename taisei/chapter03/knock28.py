#改良すべき場所ありまくり
import re
f = open('jawiki-uk.txt', 'r')
f_out = open('knock28_output.txt', 'w')
data = f.readlines()
for line in data:
    line = line.strip()
    rs = re.match(r'\|(.*?)\s\=\s*(.*?)$', line) #25 基礎情報の抽出
    if rs:
        field_name = rs.group(2)
        rs_empha_delete = re.sub(r'\'+', '', field_name) #26 強調マークアップの除去
        rs_curly_delete = re.sub(r'\{\{.*?\}\}', '', rs_empha_delete) #28 {{〜〜〜}}の除去
        rs_external_link_delete = re.sub(r'http.*?>', '', rs_curly_delete) #28 [http~~の除去
        rs_tag_delete = re.sub(r'<.*?>', '', rs_external_link_delete) #28 <br>などの除去
                
        rs2 = re.search(r'(.*)\[\[(.*?)\|(.*?)\]\](.*)', rs_tag_delete)
        if rs2: #[[ ]]の中に|がある時（wikiの内部リンクの2つ目と3つ目）
            if re.match('\[\[ファイル:', rs2.group()): #[[ファイル は内部リンクマークアップでないので分けて処理
                f_out.write(f'{rs.group(1)} : {rs_tag_delete}\n')
            else:
                field_str = rs2.group(1) + rs2.group(3) + rs2.group(4)
                field_str = re.sub(r'\[\[|\]\]', '', field_str)
                f_out.write(f'{rs.group(1)} : {field_str}\n')

        else: #wikiの内部リンクの1つめ
            rs_inter_link_delete = re.sub(r'\[\[|\]\]', '', rs_tag_delete) 
            f_out.write(f'{rs.group(1)} : {rs_inter_link_delete}\n')

f.close()
f_out.close()