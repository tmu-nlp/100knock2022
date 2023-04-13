'''
45. 動詞の格パターンの抽出：述語が取りうる格を調査するため
動詞を述語とし、動詞に係っている文節の助詞を格、述語と格を以下の仕様を満たす上で、タブ区切りで出力
・動詞を含む文節に、最左の動詞の基本形を述語とする
・述語に係る助詞を格とし、助詞が複数であれば、全ての助詞をスペース区切りで辞書順に並べる
'''

from knock41 import *



if __name__ == '__main__':
    file_path = '../data/ai.ja.txt.parsed'
    sentences = get_chunks(file_path)
    with open('./result45.txt', 'w', encoding='utf-8') as f:
        for i in range(len(sentences)):
            for chunk in sentences[i].chunks:
                for morph in chunk.morphs:
                    if morph.pos == '動詞':
                        particles = []
                        for src in chunk.srcs:
                            particles += [morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos == '助詞']
                        if len(particles) > 1:
                            particles = set(particles)
                            particles = sorted(list(particles))
                            form = ' '.join(particles)
                            print(f'{morph.base}\t{form}', file=f)