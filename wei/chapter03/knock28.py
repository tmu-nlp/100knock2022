'''
28. MediaWikiマークアップの除去
27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，国の基本情報を整形せよ．
'''

import re

def basic_dict(filename, pattern):
    with open(filename, 'r', encoding='utf-8') as f:
        dict = {}
        flag_start = False
        for line in f:
            if re.search(r'{{基礎情報\s*国',line):
                flag_start = True
                continue
            if flag_start:
                if re.search(r'^}}$', line):
                    break
            templete = re.search(pattern, line)
            if templete:
                key = templete.group(1).strip()
                dict[key] = templete.group(2).strip('')
                # print(type(dict[key]))                          # str

    return dict

# 内部リンクマークアップを除去
def remove_link(x):
    x = re.sub(r'\[\[[^\|\]]+\|[^{}\|\]]+\|([^\]]+)\]\]', r'\1', x)
    x = re.sub(r'\[\[[^\|\]]+\|([^\]]+)\]\]', r'\1', x)
    x = re.sub(r'\[\[([^\]]+)\]\]', r'\1', x)
    return x

def remove_markups(x):
    x = re.sub(r'{{.*\|.*\|([^}]*)}}', r'\1', x)
    x = re.sub(r'<([^>]*)( .*|)>.*</\1>', '', x)
    x = re.sub(r'\{\{0\}\}', '', x)
    return x

if __name__ == '__main__':

    file = 'jawiki-uk.txt'
    pattern = r'\|(.*?)\s=\s*(.+)'
    basic_info = basic_dict(file, pattern)
    dict2 = {
        key : re.sub(r"''+", '',value)
        for key, value in basic_info.items()
    }

    dict3 = {
        key : remove_link(value)
        for key, value in dict2.items()
    }

    dict4 = {
        key : remove_markups(value)
        for key, value in dict3.items()
    }
    for k, v in dict4.items():
        print(k, v)