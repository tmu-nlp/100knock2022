GPU=$1
DATA=/home/taisei/100knock/chapter10/output/knock98/finetune_prepro
MODEL=/home/taisei/100knock/chapter10/output/checkpoints_knock98_finetune/checkpoint_best.pt
BESTOUT=/home/taisei/100knock/chapter10/output/knock98/output.log
RESULTPATH=/home/taisei/100knock/chapter10/output/knock98
RAW=/home/taisei/100knock/chapter10/output/knock90_data
BLEU=/home/taisei/100knock/chapter10/output/knock98/bleu
CUDA_VISIBLE_DEVICES=$GPU fairseq-generate \
    $DATA \
    --path $MODEL \
    --batch-size 128 \
    --remove-bpe sentencepiece \
    --beam 10　> $BESTOUT

grep '^H' $BESTOUT | LC_ALL=C sort -V | cut -f3 > $RESULTPATH/system.out

CUDA_VISIBLE_DEVICES=$GPU fairseq-score \
    --sys $RESULTPATH/system.out\
    --ref $RAW/test.en > $BLEU
# 結果　
#BLEU4 = 13.24, 39.6/17.9/9.0/4.8 (BP=1.000, ratio=1.023, syslen=22562, reflen=22063)
#JParaでは1エポックしか事前学習していないけど、knock95より1.2ポイントほど上がった