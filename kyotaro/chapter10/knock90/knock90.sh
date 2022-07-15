# ja の前処理
python3 tokenizer_ja.py ./kftt-data-1.0/data/orig/kyoto-train.ja ./clean_data/kyoto-train-clean.ja
python3 tokenizer_ja.py ./kftt-data-1.0/data/orig/kyoto-dev.ja ./clean_data/kyoto-dev-clean.ja
python3 tokenizer_ja.py ./kftt-data-1.0/data/orig/kyoto-test.ja ./clean_data/kyoto-test-clean.ja

# en の前処理
python3 cleaner_en.py /home/kyotaro/100knock/kftt-data-1.0/data/orig/kyoto-train.en ./clean_data/kyoto-train-unclean.en
./tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ./clean_data/kyoto-train-unclean.en > ./clean_data/kyoto-train-clean.en
python3 cleaner_en.py /home/kyotaro/100knock/kftt-data-1.0/data/orig/kyoto-dev.en ./clean_data/kyoto-dev-unclean.en
./tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ./clean_data/kyoto-dev-unclean.en > ./clean_data/kyoto-dev-clean.en
python3 cleaner_en.py /home/kyotaro/100knock/kftt-data-1.0/data/orig/kyoto-test.en ./clean_data/kyoto-test-unclean.en
./tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ./clean_data/kyoto-test-unclean.en > ./clean_data/kyoto-test-clean.en
