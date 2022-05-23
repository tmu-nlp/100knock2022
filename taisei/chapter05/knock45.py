from knock41 import Chunk
from knock41 import Morph
from knock41 import chunk_list_list

with open("./output/knock45_output.txt", "w") as f_out:
    for chunk_list in chunk_list_list:
        for chunk_obj in chunk_list:
            for morph in chunk_obj.morphs:
                if morph.pos == "動詞":
                    case_list = [] #格のリスト
                    for src in chunk_obj.srcs: #morphが含まれるChunkオブジェクトの係元番号
                        for morph_from in chunk_list[src].morphs:
                            if morph_from.pos == "助詞":
                                case_list.append(morph_from.surface)

                    if len(case_list) != 0:
                        case_list = sorted(case_list)
                        f_out.write(f'{morph.base}\t{" ".join(case_list)}\n')
                    break #最初に出てくる動詞にしか処理しないので

#ターミナル実行コマンド
#cat knock45_output.txt | sort | uniq -c | sort -r -n | head -n 5
#cat knock45_output.txt | grep '行う' | sort | uniq -c | sort -r -n          
#cat knock45_output.txt | grep '^なる' | sort | uniq -c | sort -r -n    #^をつけないと「異なる」とかも抽出しちゃう
#cat knock45_output.txt | grep '与える' | sort | uniq -c | sort -r -n    