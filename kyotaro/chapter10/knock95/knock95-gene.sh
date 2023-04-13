GPU=$1
CUDA_VISIBLE_DEVICES=$GPU fairseq-generate /home/kyotaro/100knock/100knock2022/kyotaro/chapter10/knock95/preprocess \
    --path $models /home/kyotaro/100knock/100knock2022/kyotaro/chapter10/knock95/checkpoint/checkpoint_best.pt \
    --task translation \
    --gen-subset test \
    --batch-size 128 \
    --batch-size 128 --beam 5 | tee output.100knock.nbest.txt

grep "^H" output.100knock.nbest.txt | LC_ALL=C sort -V | cut -f3- > output.100knock.nbest.data.txt

# 翻訳結果だけを出力
cat output.100knock.nbest.data.txt | sed -r 's/(@@ )|(@@ ?$)//g' > /home/kyotaro/100knock/100knock2022/kyotaro/chapter10/knock95/output.result.txt

# BLUE
cat /home/kyotaro/100knock/100knock2022/kyotaro/chapter10/knock95/output.result.txt | sacrebleu -w 2 /home/kyotaro/100knock/100knock2022/kyotaro/chapter10/kftt-data-1.0/data/orig/kyoto-test.en


# Generate test with beam=5: BLEU4 = 12.42, 34.2/16.1/8.7/5.0 (BP=1.000, ratio=1.015, syslen=30763, reflen=30308)
# {
#  "name": "BLEU",
#  "score": 14.59,
#  "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.1.0",
#  "verbose_score": "41.8/18.1/10.0/6.0 (BP = 1.000 ratio = 1.101 hyp_len = 29231 ref_len = 26560)",
#  "nrefs": "1",
#  "case": "mixed",
#  "eff": "no",
#  "tok": "13a",
#  "smooth": "exp",
#  "version": "2.1.0"
# }