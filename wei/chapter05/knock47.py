'''
47. 機能動詞構文のマイニング
・「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
・述語に係る助詞(文節)が複数ある時、全ての助詞をスペース区切りで辞書順に並べる。
そして全ての項をスペース区切りで並べる
'''

from knock41 import *



if __name__ == '__main__':
    file_path = '../data/ai.ja.txt.parsed'
    sentences = get_chunks(file_path)
    with open('./result47.txt', 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for chunk in sentence.chunks:
                for morph in chunk.morphs:
                    if morph.pos == '動詞':
                        for src in chunk.srcs:
                            predicates = []
                            if len(sentence.chunks[src].morphs) == 2 and sentence.chunks[src].morphs[0].pos1 == 'サ変接続' and sentence.chunks[src].morphs[1].surface == 'を':   # 文節の単語数が２
                                predicates = ''.join([sentence.chunks[src].morphs[0].surface, sentence.chunks[src].morphs[1].surface, morph.base])
                                particles = []
                                arguments = []
                                for src in chunk.srcs:
                                    particles += [morph.surface for morph in sentence.chunks[src].morphs if morph.pos == '助詞']
                                    argument = ''.join([morph.surface for morph in sentence.chunks[src].morphs if morph.pos != '記号'])
                                    argument = argument.rstrip()
                                    if argument not in predicates:
                                        arguments.append(argument)

                                if len(particles) > 1:
                                    if len(arguments) > 1:
                                        particles = sorted(set(particles))
                                        arguments = sorted(set(arguments))
                                        particles_form = ' '.join(particles)
                                        arguments_form = ' '.join(arguments)
                                        predicate = ' '.join(predicates)

                                        print(f'{predicates}\t{particles_form}\t{arguments_form}', file=f)

