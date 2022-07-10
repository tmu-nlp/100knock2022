GPU=$1
DATA=/home/taisei/100knock/chapter10/output/knock90_prepro
MODEL=/home/taisei/100knock/chapter10/output/checkpoints_knock91/checkpoint_best.pt
BESTOUT=/home/taisei/100knock/chapter10/output/knock93/output.log
RESULTPATH=/home/taisei/100knock/chapter10/output/knock93
RAW=/home/taisei/100knock/chapter10/output/knock90_data
BLEU=/home/taisei/100knock/chapter10/output/knock93/bleu
CUDA_VISIBLE_DEVICES=$GPU fairseq-generate \
    $DATA \
    --path $MODEL \
    --batch-size 128 \
    --beam 5　> $BESTOUT

grep '^H' $BESTOUT | LC_ALL=C sort -V | cut -f3 > $RESULTPATH/system.out

CUDA_VISIBLE_DEVICES=$GPU fairseq-score \
    --sys $RESULTPATH/system.out\
    --ref $RAW/test.en > $BLEU
# 結果　
# BLEU4 = 9.48, 30.3/13.2/6.3/3.2 (BP=1.000, ratio=1.201, syslen=26494, reflen=22063)