# BLEUの計測
GPU=$1
DATA=/home/taisei/100knock/chapter10/output/knock95/prepro
MODEL=/home/taisei/100knock/chapter10/output/checkpoints_knock95/checkpoint_best.pt
BESTOUT=/home/taisei/100knock/chapter10/output/knock95/output_tmp.log
RESULTPATH=/home/taisei/100knock/chapter10/output/knock95
RAW=/home/taisei/100knock/chapter10/output/knock90_data
BLEU=/home/taisei/100knock/chapter10/output/knock95/bleu.txt

for beam in `seq 1 50`
do
CUDA_VISIBLE_DEVICES=$GPU fairseq-generate \
    $DATA \
    --path $MODEL \
    --batch-size 128 \
    --remove-bpe sentencepiece \
    --beam $beam　> $BESTOUT
#beam=20くらいでout of memoryしてる
grep '^H' $BESTOUT | LC_ALL=C sort -V | cut -f3 > $RESULTPATH/system.out

CUDA_VISIBLE_DEVICES=$GPU fairseq-score \
    --sys $RESULTPATH/system.out\
    --ref $RAW/test.en >> $BLEU
done

python knock95_plot.py $BLEU