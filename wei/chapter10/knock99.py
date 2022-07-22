import streamlit as st
import os
import MeCab, ipadic

# decide title name
st.title('knock99 demo')

# input
text = st.text_area('翻訳したいテキストを入力: ', max_chars=500)

start_translation=st.button("翻訳")
wakati = MeCab.Tagger('-Owakati')
res = wakati.parse(text)

with open('input.text', 'w', encoding='utf-8') as f:
    print(res, file=f)

#翻訳してファイルに結果を出力
if start_translation:
    os.system(
        "fairseq-interactive data98 \
            --path checkpoints/train.jparacrawl.lr_0.0005/checkpoint_best.pt \
            < input.txt \
            | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > knock99.txt"
    )
    with open("knock99.txt", "r", encoding='utf-8') as f:
        output_list = []
        for line in f:
            output_list.append(line)
    outputs = output_list[0]
else:
    outputs = ''

#翻訳結果を表示
st.text_area("翻訳結果: ", outputs)
