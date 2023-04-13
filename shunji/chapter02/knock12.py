with open('popular-names.txt', 'r') as rf:
    with open('col1.txt', 'w') as wf1:
        with open('col2.txt', 'w') as wf2:
            for line in rf:
                elements = line.split('\t')
                wf1.write(elements[0] + '\n')
                wf2.write(elements[1] + '\n')