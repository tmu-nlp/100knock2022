# # detoknizer
# /home/kyotaro/100knock/tools/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en < ./output/output.result.txt > 100knock_test.txt

# # 元のデータのデトークナイズ
# /home/kyotaro/100knock/tools/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en < /home/kyotaro/100knock/data/wmt17_en_de/tmp/test.de > /home/kyotaro/100knock/data/wmt17_en_de/ref/test.de

# BLUE
cat ./output/output.result.txt | sacrebleu -w 2 ./clean_data/kyoto-test-clean.en

# "score": 19.2