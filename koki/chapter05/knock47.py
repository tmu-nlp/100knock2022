## 機能動詞構文のマイニング

#動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい。
# 46のプログラムを以下の仕様を満たすように改変せよ。
from knock41 import sentences  # 解析結果の読み込み

with open('./result/output47.txt', 'w') as f:
    for sentence in sentences:
        for chunk in sentence.chunks:
            for morph in chunk.morphs:
                if morph.pos == '動詞':
                    for i, src in enumerate(chunk.srcs):
                        if (len(sentence.chunks[src].morphs) == 2) and (sentence.chunks[src].morphs[0].pos1 == 'サ変接続') and (sentence.chunks[src].morphs[1].surface == 'を'):
                            '''サ変接続名詞の発見する条件式'''
                            '''「サ変接続名詞+を(助詞)」を取得する -> (単語数が2) and (品詞細分類がサ変接続) and (表層形が'を')の条件のもとフィルタリング'''
                            predicates = f'{sentence.chunks[src].morphs[0].surface}{sentence.chunks[src].morphs[1].surface}{morph.base}'  #「サ変接続名詞 + 'を' + 動詞の基本形」 を格納(出力時の述語にあたる)
                            particles = []  # 助詞パターンの格納
                            terms = []  # 項を格納
                            tmp_terms = []
                            for src in chunk.srcs:
                                for morph_src in sentence.chunks[src].morphs:
                                    if morph_src.pos == '助詞':
                                        particles.append(morph_src.surface)  # 助詞の格納
                                    elif morph_src.pos != '記号':
                                        tmp_terms.append(morph_src.surface)  # 項の格納
                                terms.append(''.join(tmp_terms))  # 一文の項を一時格納し終えたら項を格納
                                tmp_terms = []  # 初期化

                            if len(particles) > 0:  # 助詞が発見された場合の処理
                                # 述語部(predicates): [サ変接続名詞, 'を', 動詞の基本形] --> 'サ変接続名詞を動詞の基本形'に変換, スペースは不要な点に注意
                                predicates_line = ''.join(predicates)
                                #助詞部(格)
                                particles_line = ' '.join(particles)
                                #項
                                term_line = ' '.join(terms)
                                #出力
                                result = f'{predicates_line}\t{particles_line}\t{term_line}'
                                print(result)
                                print(result, file = f)