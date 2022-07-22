#前処理
GPU=$1 \
CUDA_VISIBLE_DEVICES=$GPU fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref data/kftt-data-1.0/data/bpe/train.sub \
    --validpref data/kftt-data-1.0/data/bpe/dev.sub \
    --testpref data/kftt-data-1.0/data/bpe/test.sub \
    --destdir data/data-bin/kftt-bpe.ja-en/ \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20

#学習
GPU1=$1 \
GPU2=$2 \
CUDA_VISIBLE_DEVICES=$GPU1,$GPU2 fairseq-train data/data-bin/kftt-bpe.ja-en \
    --save-dir checkpoints/kftt-bpe.ja-en/ \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 10

#翻訳
GPU=$1
CUDA_VISIBLE_DEVICES=$GPU fairseq-interactive data/data-bin/kftt-bpe.ja-en \
    --path checkpoints/kftt-bpe.ja-en/checkpoint_best.pt \
    --remove-bpe \
    < data/kftt-data-1.0/data/bpe/test.sub.ja \
    | grep '^H' | cut -f3 > out/knock95.out

#評価
fairseq-score --sys out/knock95.out --ref data/kftt-data-1.0/data/tok/kyoto-test.en

"""
サブワード前
Namespace(ignore_case=False, order=4, ref='data/kftt-data-1.0/data/tok/kyoto-test.en', sacrebleu=False, sentence_bleu=False, sys='knock92.out')
BLEU4 = 5.34, 34.7/9.3/3.7/1.8 (BP=0.781, ratio=0.802, syslen=21432, reflen=26734) 

サブワード後
Namespace(ignore_case=False, order=4, ref='data/kftt-data-1.0/data/tok/kyoto-test.en', sacrebleu=False, sentence_bleu=False, sys='out/knock95.out')
BLEU4 = 7.25, 28.2/10.7/4.5/2.0 (BP=1.000, ratio=1.091, syslen=29154, reflen=26734)
"""
