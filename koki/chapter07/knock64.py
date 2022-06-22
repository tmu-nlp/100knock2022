with open('./questions-words.txt', 'r', encoding='utf-8_sig') as f_input, open('./results/output64.txt', 'w', encoding='utf-8_sig') as f_output:  # Winodws環境
    # with open('./questions-words.txt', 'r') as f_input, open('./results/output64.txt', 'w') as f_output:  # UNIX環境
    for line in f_input:
        line = line.split()
        if line[0] == ':':  # カテゴリを判別
            category = line[1]  # カテゴリ名を格納
        else:
            # vec[1] - vec[0] + vec[2]
            word, cos = model.most_similar(
                positive=[line[1], line[2]], negative=[line[0]], topn=1)[0]
            line = ' '.join(line)
            # テキストの可読性の観点からパイプを追加している
            f_output.write(f'{category} | {line} | {word} {str(cos)}\n')
