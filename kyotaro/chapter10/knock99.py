import MeCab
import unidic
from MeCab import Tagger
import re
import streamlit as st
import os
from fairseq.models.transformer import TransformerModel

def tokenize_ja(text):
    wakati = Tagger('-Owakati')
    num = re.compile(r'[0-9]+')
    text = num.sub('0', text)
    text = wakati.parse(text)
    return text

device = "cuda:1"

st.title("Ja-En translation")

text = st.text_area('Text to translation', '''Input Japanese sentence''')

text = tokenize_ja(text)

start_transltaion = st.button("Translation")

model = TransformerModel.from_pretrained("/home/kyotaro/100knock/100knock2022/kyotaro/chapter10/knock95/checkpoint/", "checkpoint_best.pt", "/home/kyotaro/100knock/100knock2022/kyotaro/chapter10/knock95/preprocess").to(device)

#翻訳してファイルに結果を出力
if start_transltaion:
    output = model.translate(text)
    st.text_area("", output)
