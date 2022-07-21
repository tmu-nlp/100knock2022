subword-nmt apply-bpe -c kyoto_en.codes < jparacrawl.en > train.jparacrawl.en

fairseq-preprocess -s ja -t en \
    --trainpref train.jparacrawl \
    --validpref dev.sub \
    --destdir domein  \
    --workers 20

fairseq-train domein \
    --fp16 \
    --save-dir domein_model \
    --max-epoch 3 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 98_1.log