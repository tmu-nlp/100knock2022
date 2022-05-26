from knock41 import Chunk
from knock41 import Morph
from knock41 import chunk_list_list

def get_chunk_phrase(chunk_obj2):#chunk_obj2の文節を取得
    #"".join([morph1.surface for morph1 in chunk_obj2.morphs if morph.pos != "記号"]) と同じ
    for_phrase = []
    for morph in chunk_obj2.morphs:
        if morph.pos != "記号":
            for_phrase.append(morph.surface)
    return "".join(for_phrase)

if __name__ == "__main__":
    with open("./output/knock42_output.txt", "w") as f_out:
        for chunk_list in chunk_list_list:
            for chunk_obj in chunk_list:
                if (chunk_obj.dst != -1):
                    text_from = get_chunk_phrase(chunk_obj)
                    text_for = get_chunk_phrase(chunk_list[chunk_obj.dst])
                    f_out.write(f'{text_from}\t{text_for}\n')
