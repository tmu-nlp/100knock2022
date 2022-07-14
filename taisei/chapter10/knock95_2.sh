#学習のためにバイナリデータ化
DATA=/home/taisei/100knock/chapter10/output/knock95/after_sentencepiece
echo 'binarize data'
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref $DATA/train \
    --validpref $DATA/dev \
    --testpref  $DATA/test \
    --nwordssrc 16000 \
    --nwordstgt 16000 \
    --workers 16 \
    --destdir output/knock95/prepro
