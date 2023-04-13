#!/usr/bin/env bash
# beam_sizeを変えながら、日本語を英語に翻訳
for N in `seq 5 11`; do
  fairseq-interactive --path save91/checkpoint_best.pt --beam $N data91 < test.spacy.ja | grep '^H' | cut -f3 > 94.$N.out
  done
# BLEUスコアを計算
for N in `seq 5 11`; do
  fairseq-score --sys 94.$N.out --ref test.spacy.en > 94.$N.score
  done