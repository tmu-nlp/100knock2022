from knock41 import Chunk
from knock41 import Morph
from knock41 import chunk_list_list
from knock42 import get_chunk_phrase

with open("./output/knock48_output.txt", "w") as f_out:
    for chunk_list in chunk_list_list:
        for chunk_obj1 in chunk_list:
            for morph in chunk_obj1.morphs:
                if morph.pos == "名詞":
                    ans_trans = [] #名詞を含む文節から根までへのパス
                    ans_trans.append(get_chunk_phrase(chunk_obj1))
                    chunk_obj3 = chunk_obj1 #chunk_obj1を書き換えまくるのはなんか嫌だからchunk_obj3作成

                    while (chunk_obj3.dst != -1): #係先がなくなるまで
                        chunk_obj3 = chunk_list[chunk_obj3.dst]          
                        ans_trans.append(get_chunk_phrase(chunk_obj3))
                        
                    if (len(ans_trans) != 1):
                        f_out.write(f'{" -> ".join(ans_trans)}\n')   
                    break  #chunk_obj1.morphsが名詞を複数個含む場合、同じものが何回も出ちゃうのでbreak
                