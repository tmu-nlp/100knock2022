#BLEUスコア計測
GPU=$1
CUDA_VISIBLE_DEVICES=$GPU fairseq-score --sys knock92.out --ref data/kftt-data-1.0/data/tok/kyoto-test.en
# Namespace(ignore_case=False, order=4, ref='data/kftt-data-1.0/data/tok/kyoto-test.en', sacrebleu=False, sentence_bleu=False, sys='knock92.out')
# BLEU4 = 5.34, 34.7/9.3/3.7/1.8 (BP=0.781, ratio=0.802, syslen=21432, reflen=26734) 