import pickle
import pandas as pd

with open('GoogleNews-vectors.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('./combined.csv')

sim = []
for i in range(len(df)):
    line = df.iloc[i]
    sim.append(model.similarity(line['Word 1'], line['Word 2']))

df['w2v'] = sim
result = df[['Human (mean)', 'w2v']].corr(method='spearman')
print(f'spearman corr: {result}')


'''
spearman corr:               Human (mean)       w2v
Human (mean)      1.000000  0.700017
w2v               0.700017  1.000000
'''
