import pickle
with open('GoogleNews-vectors.pkl', 'rb') as f:
    model = pickle.load(f)

# topn引数で単語数を指定
print(model.most_similar('United_States', topn=10))
