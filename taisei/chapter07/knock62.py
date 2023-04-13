from gensim.models import KeyedVectors
import pickle

if __name__ == "__main__":
    with open("./output/knock60_word2vec", "rb") as f:
        model = pickle.load(f)
    similar_10 = model.most_similar('United_States', topn=10)
    for word, value in similar_10:
        print(f'{word}\t{value}')

"""
Unites_States   0.7877248525619507
Untied_States   0.7541370987892151
United_Sates    0.7400725483894348
U.S.    0.7310774922370911
theUnited_States        0.6404394507408142
America 0.6178409457206726
UnitedStates    0.6167312264442444
Europe  0.6132988929748535
countries       0.6044804453849792
Canada  0.6019068956375122
"""