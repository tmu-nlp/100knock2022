wget http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
tar zxvf kftt-data-1.0.tar.gz
pip install -U ginza ja_ginza_electra
python3 -m spacy download en

cat kftt-data-1.0/data/orig/kyoto-train.ja | sed 's/\s+/ /g' | ginzame > train.ginza.ja
cat kftt-data-1.0/data/orig/kyoto-dev.ja | sed 's/\s+/ /g' | ginzame > dev.ginza.ja
cat kftt-data-1.0/data/orig/kyoto-test.ja | sed 's/\s+/ /g' | ginzame > test.ginza.ja

python3 knock90.py

fairseq-preprocess -s ja -t en \
    --trainpref train \
    --validpref dev \
    --destdir ja-en  \
    --workers 10 \
    --nwordssrc 15000 \
    --nwordstgt 15000