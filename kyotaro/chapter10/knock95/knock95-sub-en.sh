# 学習する
subword-nmt learn-bpe -s 16000 < kftt-data-1.0/data/orig/kyoto-train.en > kftt-data-1.0/data/code/kyoto-train-code.en
# 適応する
subword-nmt apply-bpe -c kftt-data-1.0/data/code/kyoto-train-code.en < kftt-data-1.0/data/orig/kyoto-train.en > kftt-data-1.0/data/sub/train.sub.en
subword-nmt apply-bpe -c kftt-data-1.0/data/code/kyoto-train-code.en < kftt-data-1.0/data/orig/kyoto-dev.en > kftt-data-1.0/data/sub/dev.sub.en
subword-nmt apply-bpe -c kftt-data-1.0/data/code/kyoto-train-code.en < kftt-data-1.0/data/orig/kyoto-test.en > kftt-data-1.0/data/sub/test.sub.en