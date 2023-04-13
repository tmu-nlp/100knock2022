from gensim.models import KeyedVectors
import pickle

if __name__ == "__main__":
    with open("./output/knock60_word2vec", "rb") as f:
        model = pickle.load(f)
    with open("questions-words.txt", "r") as f_data:
        data = f_data.readlines()

    with open("./output/knock64_output.txt", "w") as f_out:
        for line in data:
            line = line.strip().split()
            if len(line) == 4: #最初の行を省く
                similar_10_231 = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)
                f_out.write(f'{" ".join(line)} {similar_10_231[0][0]} {similar_10_231[0][1]}\n')

            else:
                f_out.write(f'{" ".join(line)}\n')
