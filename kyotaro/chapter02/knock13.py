with open("col1.txt") as col1:
    with open("col2.txt") as col2:
        with open("col3.txt", mode = "w") as col3:
            for name, gender in zip(col1, col2):
               name = name.strip()
               gender = gender.strip()
               col3.write(name + "\t" + gender + "\n")