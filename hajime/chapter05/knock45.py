# knock45
# 今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい．
# 動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ．
# ただし，出力は以下の仕様を満たすようにせよ．

# 1. 動詞を含む文節において，最左の動詞の基本形を述語とする
# 2. 述語に係る助詞を格とする
# 3. 述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる

# 例
# 「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」
# 動詞を含む文節において，最左の動詞の基本形 -> 作り出す（作り出した）
# 作り出すにかかる文節
# 「ジョン・マッカーシーは」
# 「会議で」
# 「用語を」
# 「作り出す で は を」という出力になる

# このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．
# 1. コーパス中で頻出する述語と格パターンの組み合わせ
# 2. 「行う」「なる」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）

# アルゴリズム
# 一文中の最左の動詞（述語）を調査
# この動詞を含む文節にかかる文節を全て選択
# 文節内の助詞を獲得
# 出力

import knock41

for sentence in knock41.sentences:  # 一文を選択
    for chunk in sentence.chunks:
        for morph1 in chunk.morphs:
            if morph1.pos == '動詞':
                part = []
                for src in chunk.srcs:
                    for morph2 in sentence.chunks[src].morphs:
                        if morph2.pos == '助詞':
                            part.append(morph2.surface)
                if len(part) > 0:
                    sort_part = sorted(list(set(part)))
                    part_line = ' '.join(sort_part)
                    line = morph1.base + "\t" + part_line
                    print(line)
                break  # このbreakで最左のみを処理している
