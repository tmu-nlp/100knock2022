import pickle
with open('GoogleNews-vectors.pkl', 'rb') as f:
    model = pickle.load(f)

print(model.similarity('United_States', 'U.S.'))
