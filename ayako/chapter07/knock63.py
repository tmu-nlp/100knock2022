# knock63
# “Spain”の単語ベクトルから”Madrid”のベクトルを引き，”Athens”のベクトルを足したベクトルを計算し，
# そのベクトルと類似度の高い10語とその類似度を出力せよ．
#　Athens：アテネらしい
# 正解はギリシャになるはず？？
from gensim.models import keyedvectors

model = keyedvectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
#パラメータpositiveは積極的に関係するキーのリストを指定，negativeは否定的に関係する(要はベクトルの引き算)キーのリストを指定
#ref:https://radimrehurek.com/gensim/models/keyedvectors.html
print(model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"], topn=10))
"""
[('Greece', 0.6898480653762817), 
('Aristeidis_Grigoriadis', 0.5606847405433655), 
('Ioannis_Drymonakos', 0.5552908778190613), 
('Greeks', 0.5450685620307922), 
('Ioannis_Christou', 0.5400863289833069), 
('Hrysopiyi_Devetzi', 0.5248444676399231), 
('Heraklio', 0.5207759737968445), 
('Athens_Greece', 0.516880989074707), 
('Lithuania', 0.5166866183280945), 
('Iraklion', 0.5146791338920593)]
"""