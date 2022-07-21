import streamlit as st
import torch
import MeCab
import unidic
from fairseq.models.transformer import TransformerModel

st.title("knock99")
wakati = MeCab.Tagger('-Owakati')

MODEL_PATH = '/home/taisei/100knock/chapter10/output/checkpoints_knock91'
DICT_PATH = '/home/taisei/100knock/chapter10/output/knock90_prepro'

model = TransformerModel.from_pretrained(MODEL_PATH, 'checkpoint_best.pt', DICT_PATH)
text = st.text_area('Text to tlansrate', '''
INPUT A TEXT
''')
start_tlans = st.button('translate')
if start_tlans:
    text_mecab = wakati.parse(str(text))
    out = model.translate(text_mecab, beam=5)
    st.text_area("", out)