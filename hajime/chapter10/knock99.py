import streamlit as st
import torch
import MeCab
import unidic
from fairseq.models.transformer import TransformerModel

# streamlit を使うとlocalhostでwebページが開ける．

st.title("knock99 : hajime")
st.write("demo")

wakati = MeCab.Tagger()

MODEL_PATH = "./"
DICT_PATH = "./ja-en-1"

model = TransformerModel.from_pretrained(
    MODEL_PATH, 'checkpoint_best_normal.pt', DICT_PATH)
text = st.text_area('Text to tlansrate', '''
翻訳したい文を入力してください．
''')
start_tlans = st.button('翻訳を実行する')

if start_tlans:
    text_mecab = wakati.parse(str(text))
    out = model.translate(text_mecab, beam=5)
    st.text_area("", out)
