## knock45 動詞の格パターンの抽出 (5)

#今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい． 
# 動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ.

from knock41 import sentences #解析結果の読み込み

with open('./result/output45.txt', 'w') as f:
    for sentence in sentences: #文
        for chunk in sentence.chunks: #文節
            for morph in chunk.morphs: #形態素
                if morph.pos == '動詞': #動詞を含む文節のみ調査
                    particles = [] #助詞の格納 (助詞...particle), 初期化
                    for src in chunk.srcs: #係り元の検査
                        for morph_src in sentence.chunks[src].morphs:#文節ループで検索中の動詞に係る助詞を検査
                            if morph_src.pos == '助詞':
                                particles.append(morph_src.surface)
                    if len(particles) > 0: #助詞が発見された場合の処理
                        #particles = sorted(particles) #sorted([list])は非破壊的、[list].sort()は破壊的処理
                        particle_line = ' '.join(particles) #空白区切りで助詞を並べる
                        result = f'{morph.base}\t{particle_line}' #結果...動詞の基本形と助詞
                        print(result)
                        print(result, file = f) #print関数のfile引数を指定してテキストに書き出し
                    else:#助詞が見つからなかった場合
                        continue
          



