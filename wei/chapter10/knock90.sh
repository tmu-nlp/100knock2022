#!/usr/bin/env bash
# preprocess/binarize the data
# src:ja trg:en;
# preprocessed data is saved to data91
fairseq-preprocess -s ja -t en \
    --trainpref train.spacy \
    --validpref dev.spacy \
    --destdir data91  \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20
# runing time:3min28s
# tokenizer=None,bpe=None,cpu=False, criterion='cross_entropy',lr_scheduler='fixed',min_loss_scale=0.0001,optimizer=None,scoring='bleu', seed=1
# [ja] Dictionary: 49320 types
# [ja] train.spacy.ja: 440288 sents, 11412336 tokens, 1.0% replaced (by <unk>)
# [ja] dev.spacy.ja: 1166 sents, 26014 tokens, 1.04% replaced (by <unk>)
# [en] Dictionary: 61944 types
# [en] train.spacy.en: 440288 sents, 11763358 tokens, 2.68% replaced (by <unk>)
# [en] dev.spacy.en: 1166 sents, 25042 tokens, 3.95% replaced (by <unk>)