from knock41 import sentences


def create_path(chunk, sentence, path_list):
    '''Chunkオブジェクト, Sentenceオブジェクトのリスト，パスを格納するリストを受け取ってパスのリストを返す関数'''
    # 係り先がないなら再帰終了
    if chunk.dst == -1:
        return path_list

    # 係り先のチャンクをpath_listの末尾に追加
    path_list.append(''.join(
        [m.surface if m.pos != '記号' else '' for m in sentence.chunks[chunk.dst].morphs]))

    # 係り先を係り元として関数に渡して再帰処理
    return create_path(sentence.chunks[chunk.dst], sentence, path_list)


sentence_path = []

for sentence in sentences:
    for chunk in sentence.chunks:
        for morph in chunk.morphs:
            if morph.pos == '名詞':
                sentence_path = [''.join([m.surface if m.pos != '記号' else '' for m in chunk.morphs])] \
                    + create_path(chunk, sentence, [])
                print(*sentence_path, sep=' -> ')
                break  # 同チャンク内に連結名詞がある場合，同じパスが表示される問題の回避
