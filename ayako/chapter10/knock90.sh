#preprocess
#ref:https://qiita.com/nymwa/items/867e05a43060d036a174
#既に単語分割されてるからそのまま使う
GPU=$1
TEXT=data/kftt-data-1.0/data/tok
CUDA_VISIBLE_DEVICES=$GPU fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $TEXT/kyoto-train \
    --validpref $TEXT/kyoto-dev \
    --testpref $TEXT/kyoto-test \
    --destdir data/data-bin/kftt.ja-en/ \
    #閾値以下の出現頻度の単語を未知語とする
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20