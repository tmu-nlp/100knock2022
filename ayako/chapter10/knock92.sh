#translation
GPU=$1
CUDA_VISIBLE_DEVICES=$GPU fairseq-interactive data/data-bin/kftt.ja-en/ \
    --path checkpoints/kftt.ja-en/checkpoint_best.pt \
    < data/kftt-data-1.0/data/tok/kyoto-test.ja \
    | grep '^H' | cut -f3 > knock92.out