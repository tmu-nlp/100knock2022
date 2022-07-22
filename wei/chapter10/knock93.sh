#!/usr/bin/env bash
fairseq-score --sys 92.out --ref test.spacy.en > 93.bleu
# Namespace(ignore_case=False, order=4, ref='test.spacy.en', sacrebleu=False, sentence_bleu=False, sys='92.out')
# BLEU4 = 5.15, 25.8/6.9/2.9/1.4 (BP=1.000, ratio=1.015, syslen=26553, reflen=26155)