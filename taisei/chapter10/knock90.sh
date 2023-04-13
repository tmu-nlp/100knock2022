wget http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
tar -zxvf kftt-data-1.0.tar.gz

mecab -Owakati kftt-data-1.0/data/orig/kyoto-dev.ja > output/knock90_data/dev.ja
mecab -Owakati kftt-data-1.0/data/orig/kyoto-train.ja > output/knock90_data/train.ja
mecab -Owakati kftt-data-1.0/data/orig/kyoto-test.ja > output/knock90_data/test.ja

cp kftt-data-1.0/data/orig/kyoto-dev.en output/knock90_data/dev.en
cp kftt-data-1.0/data/orig/kyoto-train.en output/knock90_data/train.en
cp kftt-data-1.0/data/orig/kyoto-test.en output/knock90_data/test.en

DATA=/home/taisei/100knock/chapter10/output/knock90_data
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
    --destdir output/knock90_prepro