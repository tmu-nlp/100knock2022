from gensim.models import KeyedVectors
import pickle
import pandas as pd
from scipy.stats import spearmanr

if __name__ == "__main__":
    with open("./output/knock60_word2vec", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("./combined.csv", sep=",", header=0) #Word1, Word2,Human (mean)
    df["Word vec"] = None #dfに列Word vecを追加

    for i in range(len(df)):
        df_i = df.iloc[i, :]
        simil = model.similarity(df_i["Word 1"], df_i["Word 2"])
        #df.loc[i, "Word vec"] = simil
        df.loc[i, "Word vec"] = simil
    correlation, pvalue = spearmanr(df["Human (mean)"], df["Word vec"])
    print(f'correlation = {correlation}')
    print(f'pvalue = {pvalue}')

"""
correlation = 0.7000166486272194
pvalue = 2.86866666051422e-53
"""

    
