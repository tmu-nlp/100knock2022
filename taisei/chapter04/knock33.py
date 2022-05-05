import knock30

with open("knock33_output.txt", "w") as f_out:
    for line in knock30.morpheme_line_list:
        for i in range(1, len(line) - 1):
            if line[i]["基本形"] == "の" and line[i - 1]["品詞"] == "名詞" and line[i + 1]["品詞"] == "名詞":
                f_out.write(f'{line[i - 1]["表層形"]}{line[i]["表層形"]}{line[i + 1]["表層形"]}\n')