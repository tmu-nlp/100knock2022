'''
62. 類似度の高い単語10件
“United States”とコサイン類似度が高い10語と，その類似度を出力
'''
from knock60 import *


if __name__ == '__main__':

    wv = load_wv()
    sim = wv.most_similar('United_States')   # default topn is 10

    print(sim)


    # [('Unites_States', 0.7877248525619507), ('Untied_States', 0.7541370987892151), ('United_Sates', 0.7400724291801453),
    # ('U.S.', 0.7310774326324463), ('theUnited_States', 0.6404393911361694), ('America', 0.6178410053253174),
    # ('UnitedStates', 0.6167312264442444), ('Europe', 0.6132988929748535), ('countries', 0.6044804453849792),
    #  ('Canada', 0.601906955242157)]