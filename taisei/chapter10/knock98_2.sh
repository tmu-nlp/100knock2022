#学習のためにバイナリデータ化
#JPara
DATA1=/home/taisei/100knock/chapter10/output/knock98/after_sentencepiece
echo 'binarize data'
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref $DATA1/train \
    --validpref $DATA1/dev \
    --testpref  $DATA1/test \
    --nwordssrc 16000 \
    --nwordstgt 16000 \
    --workers 16 \
    --destdir output/knock98/prepro

#KFTT
DATA2=/home/taisei/100knock/chapter10/output/knock98/finetune_after_sentencepiece
echo 'binarize data'
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref $DATA2/train \
    --validpref $DATA2/dev \
    --testpref  $DATA2/test \
    --nwordssrc 16000 \
    --nwordstgt 16000 \
    --workers 16 \
    --srcdict /home/taisei/100knock/chapter10/output/knock98/prepro/dict.ja.txt \
    --tgtdict /home/taisei/100knock/chapter10/output/knock98/prepro/dict.en.txt \
    --destdir output/knock98/finetune_prepro
