"""
係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．ただし，句読点などの記号は出力しないようにせよ．
"""

import sys

from common import Morph
from common import Chunk
from common import Sentence
from common import set_matrioshk

file_name = sys.argv[1]
sentences, chunks, morphs = set_matrioshk(file_name)

for chunk in sentences[1].chunks:
    if chunk.dst != -1:  # 係り先があることを確認
        modifer = "".join([morph.surface if morph.pos != "記号" else "" for morph in chunk.morphs])  # 係り元
        modifee = "".join([morph.surface if morph.pos != "記号" else "" for morph in sentences[1].chunks[int(chunk.dst)].morphs])  # 係り先
        print(modifer, "\t", modifee)  # タブ区切りで出力