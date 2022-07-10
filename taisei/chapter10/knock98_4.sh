#KFTTでfine-tuning
GPU1=$1
GPU2=$2

BIN=/home/taisei/100knock/chapter10/output/knock98/finetune_prepro
CUDA_VISIBLE_DEVICES=$GPU1,$GPU2 fairseq-train $BIN \
    --seed 1 \
    --keep-last-epochs 10 \
    --arch transformer \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0005 \
    --min-lr 1e-09 \
    --update-freq 8 \
    --dropout 0.1 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --save-dir output/checkpoints_knock98_finetune \
    --reset-optimizer \
    --reset-dataloader \
    --restore-file output/checkpoints_knock98/checkpoint_best.pt \
    --max-epoch 10 > knock98_train_finetune.log
