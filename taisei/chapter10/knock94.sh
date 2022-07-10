GPU=$1
DATA=/home/taisei/100knock/chapter10/output/knock90_prepro
MODEL=/home/taisei/100knock/chapter10/output/checkpoints_knock91/checkpoint_best.pt
BESTOUT=/home/taisei/100knock/chapter10/output/knock94/output_tmp.log
RESULTPATH=/home/taisei/100knock/chapter10/output/knock94
RAW=/home/taisei/100knock/chapter10/output/knock90_data
BLEU=/home/taisei/100knock/chapter10/output/knock94/bleu.txt

for beam in `seq 1 50`
do
CUDA_VISIBLE_DEVICES=$GPU fairseq-generate \
    $DATA \
    --path $MODEL \
    --batch-size 128 \
    --beam $beam　> $BESTOUT
#beam=20くらいでout of memoryしてる
grep '^H' $BESTOUT | LC_ALL=C sort -V | cut -f3 > $RESULTPATH/system.out

CUDA_VISIBLE_DEVICES=$GPU fairseq-score \
    --sys $RESULTPATH/system.out\
    --ref $RAW/test.en >> $BLEU
done

python knock94_plot.py $BLEU