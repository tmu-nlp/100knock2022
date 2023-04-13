from gensim.models import KeyedVectors
import pickle

if __name__ == "__main__":
    with open("./output/knock60_word2vec", "rb") as f:
        model = pickle.load(f)
    
    similar_10_sam = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)
    for word, value in similar_10_sam:
        print(f'{word}\t{value}')

"""
Greece  0.6898480653762817
Aristeidis_Grigoriadis  0.5606847405433655
Ioannis_Drymonakos      0.5552908778190613
Greeks  0.545068621635437
Ioannis_Christou        0.5400862693786621
Hrysopiyi_Devetzi       0.5248445272445679
Heraklio        0.5207759737968445
Athens_Greece   0.516880989074707
Lithuania       0.5166866183280945
Iraklion        0.5146791338920593
"""