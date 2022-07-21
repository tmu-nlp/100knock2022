DATA=/home/kyotaro/100knock/clean_data

echo 'binarize data'
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref $DATA/kyoto-train-clean \
    --validpref $DATA/kyoto-dev-clean \
    --testpref $DATA/kyoto-test-clean \
    --joined-dictionary \
    --destdir preprocess-kyoto