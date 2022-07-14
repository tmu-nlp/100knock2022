import MeCab
import unidic
from fairseq.models.transformer import TransformerModel

text = input('翻訳したい日本語の文を入力！')
wakati = MeCab.Tagger('-Owakati')
text_mecab = wakati.parse(text)

MODEL_PATH = '/home/taisei/100knock/chapter10/output/checkpoints_knock91'
DICT_PATH = '/home/taisei/100knock/chapter10/output/knock90_prepro'

model = TransformerModel.from_pretrained(MODEL_PATH, 'checkpoint_best.pt', DICT_PATH)
print(model.translate(text_mecab, beam=5))
"""
input : 日本の水墨画を一変させた。 
output : He changed his <unk> of Japanese ink painting in Japan.

input : 自然言語処理研究室
output : <unk> for Natural <unk>
"""
