!fairseq-train result/preprocessing/ \
  --task translation \
  --arch transformer \
  --save-dir result/train \
  --source-lang ja --target-lang en \
  --optimizer adam \
  --max-epoch 3 \
  --lr 1e-5 \ 
  --batch-size 64 \