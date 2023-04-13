import re
import requests
f = open('jawiki-uk.txt', 'r')
data = f.readlines()
basis_info = dict()
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
                basis_info[rs.group(1)] = rs_tag_delete
            else:
                field_str = rs2.group(1) + rs2.group(3) + rs2.group(4)
                field_str = re.sub(r'\[\[|\]\]', '', field_str)
                basis_info[rs.group(1)] = field_str

        else: #wikiの内部リンクの1つめ
            rs_inter_link_delete = re.sub(r'\[\[|\]\]', '', rs_tag_delete) 
            basis_info[rs.group(1)] = rs_inter_link_delete

url_api = 'https://www.mediawiki.org/w/api.php'
url_pic = basis_info['国旗画像'].replace(' ', '_')
PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles":"File:"+url_pic,
    "iiprop":"url"
}
R=requests.get(url=url_api,params=PARAMS)
DATA = R.json()
PAGES = DATA["query"]["pages"]
print(PAGES["-1"]["imageinfo"][0]["url"])

f.close()