"""
> “United States”と”U.S.”のコサイン類似度を計算せよ
"""
import pickle

model = pickle.load(open("model.pkl", "rb"))

print(model.similarity("United_States", "U.S."))
