from gensim.models import KeyedVectors

# 単語ベクトルを読み込む
m = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter07/GoogleNews-vectors-negative300.bin', binary=True)

# 類似度の高い単語とその類似度を出力する
print(m.most_similar('United_States', topn=10))

'''
[('Unites_States', 0.7877248525619507), ('Untied_States', 0.7541370987892151), 
('United_Sates', 0.7400724291801453), ('U.S.', 0.7310774326324463), 
('theUnited_States', 0.6404393911361694), ('America', 0.6178410053253174), 
('UnitedStates', 0.6167312264442444), ('Europe', 0.6132988929748535), 
('countries', 0.6044804453849792), ('Canada', 0.601906955242157)]
'''
