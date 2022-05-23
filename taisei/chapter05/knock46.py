from knock41 import Chunk
from knock41 import Morph
from knock41 import chunk_list_list
from knock42 import get_chunk_phrase

with open("./output/knock46_output.txt", "w") as f_out:
    for chunk_list in chunk_list_list:
        for chunk_obj in chunk_list:
            for morph in chunk_obj.morphs:
                if morph.pos == "動詞":
                    case_list = [] #格のリスト
                    str_list = []
                    for src in chunk_obj.srcs:
                        for morph_from in chunk_list[src].morphs:
                            if morph_from.pos == "助詞":
                                str_list.append(get_chunk_phrase(chunk_list[src]))
                                case_list.append(morph_from.surface)

                    if len(case_list) != 0:
                        pair = zip(case_list, str_list)
                        pair_sort = sorted(pair) #pair[0]でソート
                        f_out.write(f'{morph.base}\t{" ".join([v[0] for v in pair_sort])}\t{" ".join([w[1] for w in pair_sort])}\n')

                    break #最初に出てくる動詞にしか処理しないので