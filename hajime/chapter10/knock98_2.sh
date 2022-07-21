fairseq-preprocess -s ja -t en \
    --trainpref train.sub \
    --validpref dev.sub \
    --tgtdict domein/dict.en.txt \
    --srcdict domein/dict.ja.txt \
    --destdir domein_re  \
    --workers 20

fairseq-train domein_re \
    --fp16 \
    --restore-file domein_model/checkpoint3.pt \
    --save-dir re_model \
    --max-epoch 10 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 98_2.log