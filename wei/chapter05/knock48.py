'''
48. 名詞から根へのパスを抽出
「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」から以下のように抽出
ジョンマッカーシーは -> 作り出した
AIに関する -> 最初の -> 会議で -> 作り出した
最初の -> 会議で -> 作り出した
会議で -> 作り出した
人工知能という -> 用語を -> 作り出した
用語を -> 作り出した
'''
from knock41 import *



if __name__ == '__main__':
    file_path = '../data/ai.ja.txt.parsed'
    sentences = get_chunks(file_path)
    sentence = sentences[1]
    for chunk in sentence.chunks:
        for morph in chunk.morphs:
            if '名詞' in morph.pos:
                path = [''.join(morph.surface for morph in chunk.morphs if morph.pos != '記号')]
                while chunk.dst != -1:
                    path.append(''.join(morph.surface for morph in sentence.chunks[chunk.dst].morphs if morph.pos != '記号'))
                    chunk = sentence.chunks[chunk.dst]
                print('->'.join(path))
