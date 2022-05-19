from knock41 import Chunk
from knock41 import Morph
from knock41 import chunk_list_list
from knock42 import get_chunk_phrase

def get_chunk_phrase_pos(chunk_obj2, wh_pos): #品詞wh_posがchunk_obj2.morphに含まれているならchunk_obj2の文節を返す
    in_pos = False
    for morph in chunk_obj2.morphs:
        if morph.pos == wh_pos:
            in_pos = True
            break
    if in_pos:
        return get_chunk_phrase(chunk_obj2)
    else:
        return ""


if __name__ == "__main__":
    with open("./output/knock43_output.txt", "w") as f_out:
        for chunk_list in chunk_list_list:
            for chunk_obj in chunk_list:
                text_from = get_chunk_phrase_pos(chunk_obj, "名詞") #係元に名詞があるか
                text_for = ""
                if len(text_from) != 0 and chunk_obj.dst != -1: #係元に名詞がないなら走査の必要なし
                    text_for = get_chunk_phrase_pos(chunk_list[chunk_obj.dst], "動詞") #係先に動詞があるか

                if len(text_from) != 0 and len(text_for) != 0:
                    f_out.write(f'{text_from}\t{text_for}\n')
            