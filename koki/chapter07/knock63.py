import pickle
with open('GoogleNews-vectors.pkl', 'rb') as f:
    model = pickle.load(f)

# model_similar関数のpositive引数に加算する単語、negative引数に減算する単語を与える
print(model.most_similar(
    positive=['Spain', 'Athens'], negative=['Madrid'], topn=10))

'''
出力結果
[('Greece', 0.6898480653762817),
 ('Aristeidis_Grigoriadis', 0.560684859752655),
 ('Ioannis_Drymonakos', 0.5552908778190613),
 ('Greeks', 0.545068621635437),
 ('Ioannis_Christou', 0.5400862097740173),
 ('Hrysopiyi_Devetzi', 0.5248445272445679),
 ('Heraklio', 0.5207759737968445),
 ('Athens_Greece', 0.516880989074707),
 ('Lithuania', 0.5166865587234497),
 ('Iraklion', 0.5146791338920593)]
'''
