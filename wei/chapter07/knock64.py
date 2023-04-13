'''
64. アナロジーデータでの実験
単語アナロジーの評価データをダウンロードし、vec(2列目の単語)-vec(1列目の単語) + vec(3列目の単語)を計算し、そのベクトルと類似度が最も高い単語と，その類似度を求め
求めた単語と類似度は，各事例の末尾に追記
'''

import requests
from knock60 import *

# load file
# url = 'http://download.tensorflow.org/data/questions-words.txt'
# file =  requests.get(url)
# with open('../data/questions-words.txt','wb') as f:
#     f.write(file.content)



if __name__ == '__main__':
    wv = load_wv()
    with open('../data/questions-words.txt','r', encoding='utf-8') as f, open('./knock64.txt', 'a', encoding='utf-8') as f_:
        lines = f.readlines()
        for line in lines[5268:]:
            words = line.strip().split()
            if line[0] != ':':
                res = wv.most_similar(positive=[words[1], words[2]], negative=[words[0]])[0]
                f_.write(line.strip() + '\t' + str(res[0])  + '\t' + str(res[1]) + '\n')
            else:

                f_.write(line)




# 5269行目は変

'''category:
: capital-common-countries
: capital-world
: currency
: city-in-state
: family
: gram1-adjective-to-adverb
: gram2-opposite
: gram3-comparative
: gram4-superlative
: gram5-present-participle
: gram6-nationality-adjective
: gram7-past-tense
: gram8-plural
: gram9-plural-verbs
'''
