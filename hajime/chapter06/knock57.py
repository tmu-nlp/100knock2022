from sklearn.metrics import *
from knock53 import *
import numpy as np

feat = X_train.columns.values
ind = [i for i in range(1, 11)]
for c, coef in zip(lr.classes_, lr.coef_):
    top10 = pd.DataFrame(feat[np.argsort(-coef)[:10]],
                         columns=[f"top10 : {c}"], index=ind)
    worst10 = pd.DataFrame(feat[np.argsort(coef)[:10]], columns=[
                           f"worst10 : {c}"], index=ind)
    print(f"{c}'s weight")
    print(top10)
    print(worst10)
    print("---------------------")


"""
b's weight
    top10 : b
1        bank
2         fed
3         ecb
4         oil
5      stocks
6        euro
7     ukraine
8   obamacare
9       china
10      banks
   worst10 : b
1          and
2          her
3        video
4           tv
5        ebola
6        study
7       google
8        aereo
9         star
10         the
---------------------
e's weight
     top10 : e
1        chris
2   kardashian
3        movie
4         film
5      wedding
6          kim
7         fans
8         paul
9          her
10     trailer
   worst10 : e
1           us
2       update
3       google
4        study
5         says
6           gm
7          ceo
8         data
9     facebook
10       china
---------------------
m's weight
   top10 : m
1      ebola
2      study
3     cancer
4        fda
5       mers
6       drug
7     health
8    doctors
9    medical
10     heart
   worst10 : m
1          ceo
2           gm
3     facebook
4         deal
5      climate
6         bank
7        apple
8      twitter
9        sales
10      google
---------------------
t's weight
     top10 : t
1       google
2     facebook
3        apple
4      climate
5    microsoft
6           gm
7      googles
8      comcast
9   heartbleed
10        nasa
   worst10 : t
1          fed
2       health
3       cancer
4          day
5       shares
6     american
7         drug
8       stocks
9          her
10     ukraine
---------------------
"""
