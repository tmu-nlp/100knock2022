import streamlit as st
import os
import MeCab, ipadic

#タイトルを決める
st.title("knock99 Demo")

#テキストのinput
text = st.text_area("翻訳したいテキストを入力", max_chars=500)

#ボタンを押すとモデルが翻訳開始
start_translation = st.button("翻訳")

#入力を分かち書き
wakati = MeCab.Tagger("-Owakati")
result = wakati.parse(text)

#分かち書き後の入力テキストをファイル出力
with open("input.txt", "w") as f:
    print(result, file=f)

#翻訳してファイルに結果を出力
if start_translation:
    os.system(
        "fairseq-interactive data/data-bin/kftt-bpe.ja-en/ \
            --path checkpoints/kftt-bpe.ja-en/checkpoint_best.pt \
            < input.txt \
            | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > knock99.txt"
    )
    with open("knock99.txt", "r") as f:#いちいちファイル出力どうにかしたい
        output_list = []
        for line in f:
            output_list.append(line)
    outputs = output_list[0]
else:
    outputs = ""

#翻訳結果を表示
st.text_area("翻訳結果", outputs)

