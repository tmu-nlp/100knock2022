# knock62
# “United States”とコサイン類似度が高い10語と，その類似度を出力せよ．
from gensim.models import keyedvectors

model = keyedvectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
#most_similarでコサイン類似度を求められる，topnオプションで類似度上位の単語数を指定
print(model.most_similar("United_States", topn=10))

"""
[('Unites_States', 0.7877249717712402), 
('Untied_States', 0.7541371583938599), 
('United_Sates', 0.74007248878479), 
('U.S.', 0.7310774326324463), 
('theUnited_States', 0.6404393911361694), 
('America', 0.6178409457206726), 
('UnitedStates', 0.6167312264442444), 
('Europe', 0.6132988929748535), 
('countries', 0.6044804453849792), 
('Canada', 0.6019068956375122)]
"""