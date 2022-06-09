import knock30

with open("./output/knock32_output.txt", "w") as f_out:
    for line in knock30.morpheme_line_list:
        for morpheme in line:
            if morpheme["品詞"] == "動詞":
                f_out.write(f'{morpheme["基本形"]}\n')
