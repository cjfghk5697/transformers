#!/usr/bin/env bash
python run_asr.py \
--output_dir="./wav2vec2-large-lv60-100h" \
--num_train_epochs="30" \
--per_device_train_batch_size="16" \
--per_device_eval_batch_size="16" \
--eval_strategy="steps" \
--save_total_limit="3" \
--save_steps="500" \
--eval_steps="100" \
--logging_steps="50" \
--learning_rate="5e-4" \
--warmup_steps="3000" \
--model_name_or_path="facebook/wav2vec2-large-lv60" \
--fp16 \
--dataset_name="librispeech_asr" \
--dataset_config_name="clean" \
--train_split_name="train.100" \
--preprocessing_num_workers="32" \
--group_by_length \
--freeze_feature_extractor
