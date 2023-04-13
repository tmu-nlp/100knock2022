DATA=/home/kyotaro/100knock/100knock2022/kyotaro/chapter10/kftt-data-1.0/data/sub

echo 'binarize data'
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref $DATA/train.sub \
    --validpref $DATA/dev.sub \
    --testpref $DATA/test.sub \
    --joined-dictionary \
    --destdir /home/kyotaro/100knock/100knock2022/kyotaro/chapter10/knock95/preprocess