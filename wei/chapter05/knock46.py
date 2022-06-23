'''
46. 動詞の格フレーム情報を抽出
述語と格パターンに続けて項(述語に係っている文節そのもの)をタブ区切りで出力
・項は述語(動詞)に係っている文節の単語列とする
・述語に係る文節が複数ある場合、助詞と同一の基準・順序でスペース区切りで並べる
'''

from knock41 import *



if __name__ == '__main__':
    file_path = '../data/ai.ja.txt.parsed'
    sentences = get_chunks(file_path)
    with open('./result46.txt', 'w', encoding='utf-8') as f:
        for i in range(len(sentences)):
            for chunk in sentences[i].chunks:
                for morph in chunk.morphs:
                    if morph.pos == '動詞':
                        particles = []
                        arguments = []
                        for src in chunk.srcs:
                            particles += [morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos == '助詞']
                            arguments += [' '.join([morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos != '記号'])]
                        if len(particles) > 1:
                            if len(arguments) > 1:
                                particles = sorted(set(particles))
                                arguments = sorted(set(arguments))
                                particles_form = ' '.join(particles)
                                arguments_form = ' '.join(arguments)

                                print(f'{morph.base}\t{particles_form}\t{arguments}', file=f)


