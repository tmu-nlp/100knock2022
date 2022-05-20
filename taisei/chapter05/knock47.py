from knock41 import Chunk
from knock41 import Morph
from knock41 import chunk_list_list
from knock42 import get_chunk_phrase

with open("./output/knock47_output.txt", "w") as f_out:
    for chunk_list in chunk_list_list:
        for chunk_obj in chunk_list:
            for morph in chunk_obj.morphs:
                if morph.pos == "動詞":
                    for src in chunk_obj.srcs:
                         #サ変名詞＋を のみで構成されるChunkを探す
                        if len(chunk_list[src].morphs) == 2 and (chunk_list[src].morphs[0].pos1 == "サ変接続") and (chunk_list[src].morphs[1].surface == "を"):
                            ans_str = f'{chunk_list[src].morphs[0].surface}{chunk_list[src].morphs[1].surface}{morph.base}'   
                            ans_src = src #ans_strの文節番号を保持
                            case_list = [] #格のリスト
                            str_list = []
                            for src5 in chunk_obj.srcs:
                                if src5 == ans_src:#ans_strの文節番号が同じ時、ans_strと同じものなので処理しない
                                    continue 
                                for morph_from in chunk_list[src5].morphs:
                                    if morph_from.pos == "助詞":
                                        str_list.append(get_chunk_phrase(chunk_list[src5]))
                                        case_list.append(morph_from.surface)

                            if len(case_list) != 0 and len(ans_str) != 0:
                                pair = zip(case_list, str_list)
                                pair = sorted(pair)
                                f_out.write(f'{ans_str}\t{" ".join([v[0] for v in pair])}\t{" ".join([w[1] for w in pair])}\n')

                            # elif len(ans_str) != 0: #述語に係る助詞がひとつしかない場合（case_list,str_listは空の時）
                            #     f_out.write(f'{ans_str}\t\n')
              
                    break