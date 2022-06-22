from gensim.models import KeyedVectors
m = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter07/GoogleNews-vectors-negative300.bin', binary=True)

with open('./100knock2022/DUAN/chapter07/questions-words.txt') as input, open('./100knock2022/DUAN/chapter07/output.txt', 'w') as output:
    for line in input:
        line = line.split()
        if line[0] == ':':
            category = line[1]
        else:
            word, cos = m.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)[0]
            output.write(' '.join([category] + line + [word, str(cos) + '\n']))