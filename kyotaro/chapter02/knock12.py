with open("popular-names.txt") as f:
    with open("col1.txt", mode = "w") as name:
        with open("col2.txt", mode = "w") as gender:
            for line in f:
                word = line.split("\t")
                name.write(word[0])
                name.write("\n")
                gender.write(word[1])
                gender.write("\n")
        