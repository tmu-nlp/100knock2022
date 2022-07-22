#!/usr/bin/env bash

# pip install subword-nmt
subword-nmt learn-bpe -s 16000 < kftt-data-1.0/data/orig/kyoto-train.en > kyoto_en.codes
subword-nmt apply-bpe -c kyoto_en.codes < kftt-data-1.0/data/orig/kyoto-train.en > train.sub.en
subword-nmt apply-bpe -c kyoto_en.codes < kftt-data-1.0/data/orig/kyoto-dev.en > dev.sub.en
subword-nmt apply-bpe -c kyoto_en.codes < kftt-data-1.0/data/orig/kyoto-test.en > test.sub.en
# runing time:1min56sec
# vocab_size:16000

fairseq-preprocess -s ja -t en\
    --trainpref train.sub \
    --validpref dev.sub \
    --testpref test.sub \
    --tokenizer space \
    --workers 20 \
    --thresholdsrc 3 \
    --thresholdtgt 3 \
    --task translation \
    --workers 20 \
    --destdir knock95_subwords_sp

# Namespace(aim_repo=None, aim_run_hash=None, align_suffix=None, alignfile=None, all_gather_list_size=16384, amp=False, amp_batch_retries=2, amp_init_scale=128, amp_scale_window=None, azureml_logging=False, bf16=False, bpe=None, cpu=False, criterion='cross_entropy', dataset_impl='mmap', destdir='knock95_subwords_sp', dict_only=False, empty_cache_freq=0, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, joined_dictionary=False, log_file=None, log_format=None, log_interval=100, lr_scheduler='fixed', memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, model_parallel_size=1, no_progress_bar=False, nwordssrc=-1, nwordstgt=-1, on_cpu_convert_precision=False, only_source=False, optimizer=None, padding_factor=8, plasma_path='/tmp/plasma', profile=False, quantization_config_path=None, reset_logging=False, scoring='bleu', seed=1, source_lang='ja', srcdict=None, suppress_crashes=False, target_lang='en', task='translation', tensorboard_logdir=None, testpref='test.sub', tgtdict=None, threshold_loss_scale=None, thresholdsrc=3, thresholdtgt=3, tokenizer='space', tpu=False, trainpref='train.sub', use_plasma_view=False, user_dir=None, validpref='dev.sub', wandb_project=None, workers=20)
# [ja] Dictionary: 17048 types
# [ja] train.sub.ja: 440288 sents, 10462018 tokens, 0.0147% replaced (by <unk>)
# [ja] Dictionary: 17048 types
# [ja] dev.sub.ja: 1166 sents, 24223 tokens, 0.0206% replaced (by <unk>)
# [ja] Dictionary: 17048 types
# [ja] test.sub.ja: 1160 sents, 26130 tokens, 0.0153% replaced (by <unk>)
# [en] Dictionary: 18656 types
# [en] train.sub.en: 440288 sents, 13280091 tokens, 0.022% replaced (by <unk>)
# [en] Dictionary: 18656 types
# [en] dev.sub.en: 1166 sents, 29011 tokens, 0.0103% replaced (by <unk>)
# [en] Dictionary: 18656 types
# [en] test.sub.en: 1160 sents, 31468 tokens, 0.0254% replaced (by <unk>)
# Wrote preprocessed data to knock95_subwords_sp


GPU_1=$1 \
GPU_2=$2 \
CUDA_VISIBLE_DEVICES=$GPU_1,$GPU_2 \
fairseq-train knock95_subwords_sp \
    --fp16 \
    --save-dir save95 \
    --max-epoch 3 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --update-freq 1 \
    --dropout 0.2 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 95.log

fairseq-interactive --path save95/checkpoint_best.pt knock95_subwords_sp < test.spacy.ja | grep '^H' | cut -f3 > 95.out
# [ja] dictionary: 17048 types
# [en] dictionary: 18656 types
# Total time: 379.424 seconds; translation time: 367.011

fairseq-score --sys 95.out.spacy --ref test.spacy.en
# Namespace(ignore_case=False, order=4, ref='test.spacy.en', sacrebleu=False, sentence_bleu=False, sys='95.out.spacy')
# BLEU4 = 1.70, 16.7/2.8/0.7/0.3 (BP=1.000, ratio=1.350, syslen=35302, reflen=26155)
# ??? sth wrong must have happened


