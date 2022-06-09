import knock30

with open("./output/knock34_output.txt", "w") as f_out:
    for line in knock30.morpheme_line_list:
        noun = ""
        cnt = 0
        for morpheme in line:
            if morpheme["品詞"] == "名詞":
                noun += morpheme["表層形"]
                cnt += 1
            elif cnt >= 2: #名詞が2つ以上連結してる時
                f_out.write(f'{noun}\n')
                cnt = 0
                noun = ""

            else:#名詞がひとつ以下のとき（連結名詞にならない時）
                noun = ""
                cnt = 0
            
