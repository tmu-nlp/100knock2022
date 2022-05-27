## knock46.py

#45のプログラムを改変し，述語と格パターンに続けて項（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．

from knock41 import sentences #解析結果の読み込み

with open('./result/output46.txt', 'w') as f:
    for sentence in sentences:
        for chunk in sentence.chunks:
            for morph in chunk.morphs:
                if morph.pos == '動詞':
                    particles = [] #格パターンを格納
                    tmp_terms = [] #項を一時格納
                    terms = []#項を格納
                    for src in chunk.srcs:
                        for morph_src in sentence.chunks[src].morphs:
                            if morph_src.pos == '助詞':
                                particles.append(morph_src.surface)#助詞を格納
                            
                            elif morph_src.pos != '記号':
                                tmp_terms.append(morph_src.surface)#項を格納
                        terms.append(''.join(tmp_terms))#一文の項を一時格納し終えたら項を格納
                        tmp_terms = []#初期化

                    if len(particles) > 0: #助詞が発見された場合の処理
                        #particles = sorted(particles)
                        particle_line = ' '.join(particles)#格納した助詞を空白区切り
                        term_line = ' '.join(terms)#項
                        result = f'{morph.base}\t{particle_line}\t{term_line}'#結果...動詞の基本形、助詞、項
                        print(result)
                        print(result, file = f)