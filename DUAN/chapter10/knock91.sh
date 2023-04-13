fairseq-preprocess -s ja -t en \
    --trainpref train.spacy \
    --validpref dev.spacy \
    --destdir data91  \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20

:<<!
| INFO | fairseq_cli.preprocess | Namespace(no_progress_bar=False, log_interval=100, log_format=None, log_file=None, aim_repo=None, aim_run_hash=None, tensorboard_logdir=None, wandb_project=None, azureml_logging=False, seed=1, cpu=False, tpu=False, bf16=False, memory_efficient_bf16=False, fp16=False, memory_efficient_fp16=False, fp16_no_flatten_grads=False, fp16_init_scale=128, fp16_scale_window=None, fp16_scale_tolerance=0.0, on_cpu_convert_precision=False, min_loss_scale=0.0001, threshold_loss_scale=None, amp=False, amp_batch_retries=2, amp_init_scale=128, amp_scale_window=None, user_dir=None, empty_cache_freq=0, all_gather_list_size=16384, model_parallel_size=1, quantization_config_path=None, profile=False, reset_logging=False, suppress_crashes=False, use_plasma_view=False, plasma_path='/tmp/plasma', criterion='cross_entropy', tokenizer=None, bpe=None, optimizer=None, lr_scheduler='fixed', scoring='bleu', task='translation', source_lang='ja', target_lang='en', trainpref='train.spacy', validpref='dev.spacy', testpref=None, align_suffix=None, destdir='data91', thresholdtgt=5, thresholdsrc=5, tgtdict=None, srcdict=None, nwordstgt=-1, nwordssrc=-1, alignfile=None, dataset_impl='mmap', joined_dictionary=False, only_source=False, padding_factor=8, workers=20, dict_only=False)
| INFO | fairseq_cli.preprocess | [ja] Dictionary: 53928 types
| INFO | fairseq_cli.preprocess | [ja] train.spacy.ja: 440288 sents, 11547511 tokens, 1.24% replaced (by <unk>)
| INFO | fairseq_cli.preprocess | [ja] Dictionary: 53928 types
| INFO | fairseq_cli.preprocess | [ja] dev.spacy.ja: 1166 sents, 26468 tokens, 1.24% replaced (by <unk>)
| INFO | fairseq_cli.preprocess | [en] Dictionary: 55472 types
| INFO | fairseq_cli.preprocess | [en] train.spacy.en: 440288 sents, 12321438 tokens, 1.56% replaced (by <unk>)
| INFO | fairseq_cli.preprocess | [en] Dictionary: 55472 types
| INFO | fairseq_cli.preprocess | [en] dev.spacy.en: 1166 sents, 26101 tokens, 2.83% replaced (by <unk>)
| INFO | fairseq_cli.preprocess | Wrote preprocessed data to data91
!

fairseq-train data91 \
    --fp16 \
    --save-dir save91 \
    --max-epoch 10 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --update-freq 1 \
    --dropout 0.2 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 91.log

:<<!
epoch 001: 100% 1822/1823 [10:27<00:00,  2.97it/s, loss=7.527, nll_loss=6.232, ppl=75.16, wps=19156.5, ups=2.92, wpb=6565.9, bsz=219.4, num_updates=1800, lr=0.0009, gnorm=0.873, clip=22, loss_scale=8, train_wall=34, wall=622]
epoch 001 | valid on 'valid' subset:   0% 0/7 [00:00<?, ?it/s]
epoch 001 | valid on 'valid' subset:  14% 1/7 [00:00<00:00,  6.44it/s]
epoch 001 | valid on 'valid' subset:  29% 2/7 [00:00<00:00,  7.96it/s]
epoch 001 | valid on 'valid' subset:  43% 3/7 [00:00<00:00,  8.64it/s]
epoch 001 | valid on 'valid' subset:  57% 4/7 [00:00<00:00,  8.96it/s]
epoch 001 | valid on 'valid' subset:  71% 5/7 [00:00<00:00,  9.13it/s]
epoch 001 | valid on 'valid' subset:  86% 6/7 [00:00<00:00,  8.75it/s]
epoch 001 | valid on 'valid' subset: 100% 7/7 [00:00<00:00,  8.80it/s]
epoch 002: 100% 1822/1823 [10:29<00:00,  2.81it/s, loss=6.681, nll_loss=5.262, ppl=38.36, wps=19541.1, ups=2.9, wpb=6729.2, bsz=253.8, num_updates=3600, lr=0.000745356, gnorm=0.608, clip=7, loss_scale=8, train_wall=34, wall=1261]
epoch 002 | valid on 'valid' subset:   0% 0/7 [00:00<?, ?it/s]
epoch 002 | valid on 'valid' subset:  14% 1/7 [00:00<00:00,  6.20it/s]
epoch 002 | valid on 'valid' subset:  29% 2/7 [00:00<00:00,  7.78it/s]
epoch 002 | valid on 'valid' subset:  43% 3/7 [00:00<00:00,  8.48it/s]
epoch 002 | valid on 'valid' subset:  57% 4/7 [00:00<00:00,  8.84it/s]
epoch 002 | valid on 'valid' subset:  71% 5/7 [00:00<00:00,  9.06it/s]
epoch 002 | valid on 'valid' subset:  86% 6/7 [00:00<00:00,  8.69it/s]
epoch 002 | valid on 'valid' subset: 100% 7/7 [00:00<00:00,  8.80it/s]
epoch 003: 100% 1822/1823 [10:28<00:00,  3.04it/s, loss=6.31, nll_loss=4.831, ppl=28.45, wps=19885.3, ups=2.85, wpb=6965.5, bsz=264.6, num_updates=5400, lr=0.000608581, gnorm=0.565, clip=5, loss_scale=4, train_wall=35, wall=1901]
epoch 003 | valid on 'valid' subset:   0% 0/7 [00:00<?, ?it/s]
epoch 003 | valid on 'valid' subset:  14% 1/7 [00:00<00:01,  5.72it/s]
epoch 003 | valid on 'valid' subset:  29% 2/7 [00:00<00:00,  7.56it/s]
epoch 003 | valid on 'valid' subset:  43% 3/7 [00:00<00:00,  8.23it/s]
epoch 003 | valid on 'valid' subset:  57% 4/7 [00:00<00:00,  8.70it/s]
epoch 003 | valid on 'valid' subset:  71% 5/7 [00:00<00:00,  9.05it/s]
epoch 003 | valid on 'valid' subset:  86% 6/7 [00:00<00:00,  8.72it/s]
epoch 003 | valid on 'valid' subset: 100% 7/7 [00:00<00:00,  8.86it/s]
epoch 004: 100% 1822/1823 [10:27<00:00,  3.05it/s, loss=5.993, nll_loss=4.464, ppl=22.06, wps=19946.9, ups=2.89, wpb=6907.6, bsz=247.1, num_updates=7200, lr=0.000527046, gnorm=0.574, clip=3, loss_scale=4, train_wall=34, wall=2540]
epoch 004 | valid on 'valid' subset:   0% 0/7 [00:00<?, ?it/s]
epoch 004 | valid on 'valid' subset:  14% 1/7 [00:00<00:00,  6.15it/s]
epoch 004 | valid on 'valid' subset:  29% 2/7 [00:00<00:00,  7.60it/s]
epoch 004 | valid on 'valid' subset:  43% 3/7 [00:00<00:00,  8.34it/s]
epoch 004 | valid on 'valid' subset:  57% 4/7 [00:00<00:00,  8.80it/s]
epoch 004 | valid on 'valid' subset:  71% 5/7 [00:00<00:00,  9.12it/s]
epoch 004 | valid on 'valid' subset:  86% 6/7 [00:00<00:00,  8.80it/s]
epoch 004 | valid on 'valid' subset: 100% 7/7 [00:00<00:00,  8.92it/s]
epoch 005: 100% 1822/1823 [10:27<00:00,  2.92it/s, loss=5.84, nll_loss=4.284, ppl=19.49, wps=19690.1, ups=2.9, wpb=6798.9, bsz=266.3, num_updates=9100, lr=0.000468807, gnorm=0.593, clip=4, loss_scale=4, train_wall=34, wall=3207]
epoch 005 | valid on 'valid' subset:   0% 0/7 [00:00<?, ?it/s]
epoch 005 | valid on 'valid' subset:  14% 1/7 [00:00<00:01,  5.89it/s]
epoch 005 | valid on 'valid' subset:  29% 2/7 [00:00<00:00,  7.58it/s]
epoch 005 | valid on 'valid' subset:  43% 3/7 [00:00<00:00,  8.37it/s]
epoch 005 | valid on 'valid' subset:  57% 4/7 [00:00<00:00,  8.76it/s]
epoch 005 | valid on 'valid' subset:  71% 5/7 [00:00<00:00,  9.00it/s]
epoch 005 | valid on 'valid' subset:  86% 6/7 [00:00<00:00,  8.66it/s]
epoch 005 | valid on 'valid' subset: 100% 7/7 [00:00<00:00,  8.75it/s]
epoch 006: 100% 1822/1823 [10:27<00:00,  2.94it/s, loss=5.696, nll_loss=4.117, ppl=17.35, wps=19757.3, ups=2.95, wpb=6707, bsz=219.3, num_updates=10900, lr=0.000428353, gnorm=0.6, clip=4, loss_scale=2, train_wall=34, wall=3846]
epoch 006 | valid on 'valid' subset:   0% 0/7 [00:00<?, ?it/s]
epoch 006 | valid on 'valid' subset:  14% 1/7 [00:00<00:00,  6.23it/s]
epoch 006 | valid on 'valid' subset:  29% 2/7 [00:00<00:00,  7.75it/s]
epoch 006 | valid on 'valid' subset:  43% 3/7 [00:00<00:00,  8.47it/s]
epoch 006 | valid on 'valid' subset:  57% 4/7 [00:00<00:00,  8.89it/s]
epoch 006 | valid on 'valid' subset:  71% 5/7 [00:00<00:00,  9.14it/s]
epoch 006 | valid on 'valid' subset:  86% 6/7 [00:00<00:00,  8.70it/s]
epoch 006 | valid on 'valid' subset: 100% 7/7 [00:00<00:00,  8.72it/s]
epoch 007: 100% 1822/1823 [10:26<00:00,  2.93it/s, loss=5.65, nll_loss=4.062, ppl=16.7, wps=19899.8, ups=2.9, wpb=6871.2, bsz=257, num_updates=12700, lr=0.000396838, gnorm=0.648, clip=4, loss_scale=2, train_wall=34, wall=4485]
epoch 007 | valid on 'valid' subset:   0% 0/7 [00:00<?, ?it/s]
epoch 007 | valid on 'valid' subset:  14% 1/7 [00:00<00:00,  6.27it/s]
epoch 007 | valid on 'valid' subset:  29% 2/7 [00:00<00:00,  7.84it/s]
epoch 007 | valid on 'valid' subset:  43% 3/7 [00:00<00:00,  8.57it/s]
epoch 007 | valid on 'valid' subset:  57% 4/7 [00:00<00:00,  8.96it/s]
epoch 007 | valid on 'valid' subset:  71% 5/7 [00:00<00:00,  9.19it/s]
epoch 007 | valid on 'valid' subset:  86% 6/7 [00:00<00:00,  8.70it/s]
epoch 007 | valid on 'valid' subset: 100% 7/7 [00:00<00:00,  8.76it/s]
epoch 008: 100% 1822/1823 [10:26<00:00,  2.98it/s, loss=5.611, nll_loss=4.015, ppl=16.17, wps=19423.4, ups=2.87, wpb=6761.2, bsz=264.8, num_updates=14500, lr=0.000371391, gnorm=0.658, clip=4, loss_scale=2, train_wall=35, wall=5123]
epoch 008 | valid on 'valid' subset:   0% 0/7 [00:00<?, ?it/s]
epoch 008 | valid on 'valid' subset:  14% 1/7 [00:00<00:00,  6.19it/s]
epoch 008 | valid on 'valid' subset:  29% 2/7 [00:00<00:00,  7.72it/s]
epoch 008 | valid on 'valid' subset:  43% 3/7 [00:00<00:00,  8.41it/s]
epoch 008 | valid on 'valid' subset:  57% 4/7 [00:00<00:00,  8.74it/s]
epoch 008 | valid on 'valid' subset:  71% 5/7 [00:00<00:00,  9.05it/s]
epoch 008 | valid on 'valid' subset:  86% 6/7 [00:00<00:00,  8.71it/s]
epoch 008 | valid on 'valid' subset: 100% 7/7 [00:00<00:00,  8.83it/s]
epoch 009: 100% 1822/1823 [10:26<00:00,  2.89it/s, loss=5.406, nll_loss=3.779, ppl=13.73, wps=19933, ups=2.91, wpb=6846.1, bsz=236.7, num_updates=16400, lr=0.000349215, gnorm=0.635, clip=3, loss_scale=2, train_wall=34, wall=5796]
epoch 009 | valid on 'valid' subset:   0% 0/7 [00:00<?, ?it/s]
epoch 009 | valid on 'valid' subset:  14% 1/7 [00:00<00:00,  6.10it/s]
epoch 009 | valid on 'valid' subset:  29% 2/7 [00:00<00:00,  7.70it/s]
epoch 009 | valid on 'valid' subset:  43% 3/7 [00:00<00:00,  8.43it/s]
epoch 009 | valid on 'valid' subset:  57% 4/7 [00:00<00:00,  8.74it/s]
epoch 009 | valid on 'valid' subset:  71% 5/7 [00:00<00:00,  8.97it/s]
epoch 009 | valid on 'valid' subset:  86% 6/7 [00:00<00:00,  8.66it/s]
epoch 009 | valid on 'valid' subset: 100% 7/7 [00:00<00:00,  8.82it/s]
epoch 010: 100% 1822/1823 [10:26<00:00,  2.85it/s, loss=5.305, nll_loss=3.661, ppl=12.65, wps=19498, ups=2.93, wpb=6656.2, bsz=252.3, num_updates=18200, lr=0.000331497, gnorm=0.651, clip=3, loss_scale=2, train_wall=34, wall=6426]
epoch 010 | valid on 'valid' subset:   0% 0/7 [00:00<?, ?it/s]
epoch 010 | valid on 'valid' subset:  14% 1/7 [00:00<00:01,  5.78it/s]
epoch 010 | valid on 'valid' subset:  29% 2/7 [00:00<00:00,  7.55it/s]
epoch 010 | valid on 'valid' subset:  43% 3/7 [00:00<00:00,  8.34it/s]
epoch 010 | valid on 'valid' subset:  57% 4/7 [00:00<00:00,  8.81it/s]
epoch 010 | valid on 'valid' subset:  71% 5/7 [00:00<00:00,  9.10it/s]
epoch 010 | valid on 'valid' subset:  86% 6/7 [00:00<00:00,  8.64it/s]
epoch 010 | valid on 'valid' subset: 100% 7/7 [00:00<00:00,  8.71it/s]
!