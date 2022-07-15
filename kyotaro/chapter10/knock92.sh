GPU=$1
CUDA_VISIBLE_DEVICES=$GPU fairseq-generate /home/kyotaro/100knock/preprocess-kyoto \
    --path $models /home/kyotaro/100knock/checkpoints-kyoto/checkpoint_best.pt \
    --task translation \
    --gen-subset test \
    --batch-size 128 \
    --batch-size 128 --beam 5 | tee output.100knock.nbest.txt

grep "^H" output.100knock.nbest.txt | LC_ALL=C sort -V | cut -f3- > output.100knock.nbest.data.txt

# 翻訳結果だけを出力
cat output.100knock.nbest.data.txt | sed -r 's/(@@ )|(@@ ?$)//g' > output.result.txt