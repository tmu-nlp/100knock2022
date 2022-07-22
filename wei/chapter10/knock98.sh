#!/usr/bin/env bash
# execute subword
subword-nmt apply-bpe -c kyoto_en.codes < jparacrawl.en > train.jparacrawl.en

fairseq-preprocess -s ja -t en \
    --trainpref train.jparacrawl \
    --validpref dev.sub \
    --destdir data98  \
    --workers 20


#学習
# training with different dropout rate
for N in `seq 0.001 0.0005`; do
  fairseq-train data98 \
        --fp16
        --save-dir checkpoints/train.jparacrawl.lr_$N \
        --arch transformer --share-decoder-input-output-embed \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr $N --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --dropout $N --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 4096 \
        --max-epoch 5
  done

#推論

for N in `seq 0.001 0.0005` ; do
  fairseq-interactive data98 \
    --path checkpoints/train.jparacrawl.lr_$N/checkpoint_best.pt \
    < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > out98/lr_$N.out
  done


#BLEUスコアを計算
for N in `seq 0.001 0.0005` ; do
    echo lr=$N >> out98/score98.out
    fairseq-score --sys out98/lr_$N.out --ref test.spacy.en >> out98/score98.out
  done

