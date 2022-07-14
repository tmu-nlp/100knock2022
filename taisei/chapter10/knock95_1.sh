#senetncepieceを使ってサブワード化をする
#sentencepieceモデルの作成
spm_train \
    --input=/home/taisei/100knock/chapter10/output/knock90_data/train.ja \
    --model_prefix=sentencepiece_model.ja \
    --vocab_size=16000 \
    --character_coverage=0.9995 \
    --model_type=unigram

spm_train \
    --input=/home/taisei/100knock/chapter10/output/knock90_data/train.en \
    --model_prefix=sentencepiece_model.en \
    --vocab_size=16000 \
    --character_coverage=1.0 \
    --model_type=unigram

#上で学習したsentencepieceモデルを使って、訓練データ評価データ開発データをサブワード分割
spm_encode --model=sentencepiece_model.en.model --output_format=piece < /home/taisei/100knock/chapter10/output/knock90_data/train.en > /home/taisei/100knock/chapter10/output/knock95/after_sentencepiece/train.en
spm_encode --model=sentencepiece_model.en.model --output_format=piece < /home/taisei/100knock/chapter10/output/knock90_data/dev.en > /home/taisei/100knock/chapter10/output/knock95/after_sentencepiece/dev.en
spm_encode --model=sentencepiece_model.en.model --output_format=piece < /home/taisei/100knock/chapter10/output/knock90_data/test.en > /home/taisei/100knock/chapter10/output/knock95/after_sentencepiece/test.en

spm_encode --model=sentencepiece_model.ja.model --output_format=piece < /home/taisei/100knock/chapter10/output/knock90_data/train.ja > /home/taisei/100knock/chapter10/output/knock95/after_sentencepiece/train.ja
spm_encode --model=sentencepiece_model.ja.model --output_format=piece < /home/taisei/100knock/chapter10/output/knock90_data/dev.ja > /home/taisei/100knock/chapter10/output/knock95/after_sentencepiece/dev.ja
spm_encode --model=sentencepiece_model.ja.model --output_format=piece < /home/taisei/100knock/chapter10/output/knock90_data/test.ja > /home/taisei/100knock/chapter10/output/knock95/after_sentencepiece/test.ja
