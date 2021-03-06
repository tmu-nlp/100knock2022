#!/usr/bin/env bash
#学習
# training with different dropout rate
for N in `seq 0.1 0.2 0.5`; do
  fairseq-train knock95_subwords_sp \
        --save-dir checkpoints/train.sub.dropout_$N \
        --arch transformer --share-decoder-input-output-embed \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --dropout $N --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 4096 \
        --max-epoch 5
  done

#推論
for N in `seq 0.1 0.2 0.5` ; do
  fairseq-interactive knock95_subwords_sp \
    --path checkpoints/train.sub.dropout_$N/checkpoint_best.pt \
    < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > out97/dropout_$N.out
  done

#BLEUスコアを計算
for N in `seq 0.1 0.2 0.5` ; do
    echo dropout=$N >> out97/score97.out
    fairseq-score --sys out97/dropout_$N.out --ref test.spacy.en >> out97/score97.out
  done
