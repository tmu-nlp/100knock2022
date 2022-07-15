GPU=$1
for i in 1 2 4 8 16; do
    CUDA_VISIBLE_DEVICES=$GPU fairseq-generate /home/kyotaro/100knock/preprocess-kyoto \
        --path $models /home/kyotaro/100knock/checkpoints-kyoto/checkpoint_best.pt \
        --task translation \
        --gen-subset test \
        --batch-size 32 \
        --batch-size 32 --beam $i | tee ./output/output.100knock.nbest.$i.txt
    grep "^H" ./output/output.100knock.nbest.$i.txt | LC_ALL=C sort -V | cut -f3- > ./output/output.100knock.nbest.$i.data.txt

    # 翻訳結果だけを出力
    cat ./output/output.100knock.nbest.$i.data.txt | sed -r 's/(@@ )|(@@ ?$)//g' > ./output/output.result.$i.txt
done

for i in 1 2 4 8 16; do
    cat ./output/output.result.$i.txt | sacrebleu -w 2 --force ./clean_data/kyoto-test-clean.en
done