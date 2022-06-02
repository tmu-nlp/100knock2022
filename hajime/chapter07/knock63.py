from gensim.models import KeyedVectors
import pickle

with open("word2vec.pkl", "rb") as f:
    model = pickle.load(f)

print(model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"]))

"""
[('Greece', 0.6898480653762817), 
('Aristeidis_Grigoriadis', 0.560684859752655), 
('Ioannis_Drymonakos', 0.555290937423706),
('Greeks', 0.5450686812400818), 
('Ioannis_Christou', 0.5400862693786621), 
('Hrysopiyi_Devetzi', 0.5248445272445679), 
('Heraklio', 0.5207759141921997), 
('Athens_Greece', 0.5168809294700623), 
('Lithuania', 0.5166866183280945), 
('Iraklion', 0.5146791934967041)]
"""
