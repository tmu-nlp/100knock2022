#beam search
GPU=$1
# for N in `seq 1 20` ; do
#     CUDA_VISIBLE_DEVICES=$GPU fairseq-interactive \
#     --path checkpoints/kftt.ja-en/checkpoint_best.pt \
#     --beam $N data/data-bin/kftt.ja-en/ \
#     < data/kftt-data-1.0/data/tok/kyoto-dev.ja | grep '^H' | cut -f3 > out94/beam.$N.out
# done

for N in `seq 1 20` ; do
    echo beam=$N >> out94/score94.out 
    CUDA_VISIBLE_DEVICES=$GPU fairseq-score --sys out94/beam.$N.out --ref data/kftt-data-1.0/data/tok/kyoto-dev.en >> out94/score94.out
done