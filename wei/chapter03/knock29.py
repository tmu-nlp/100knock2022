from knock28 import *
import re
import requests

def get_url(text):
    url_file = text['国旗画像'].replace(' ', '_')
    url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + url_file + '&prop=imageinfo&iiprop=url&format=json'
    data = requests.get(url)
    return re.search(r'"url":"(.+?)"', data.text).group(1)


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

    res = get_url(dict4)

    print(res)