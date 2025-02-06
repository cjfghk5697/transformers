#!/usr/bin/env bash
python run_common_voice.py \
    --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
    --dataset_config_name="tr" \
    --output_dir=./wav2vec2-large-xlsr-turkish-demo \
    --overwrite_output_dir \
    --num_train_epochs="5" \
    --per_device_train_batch_size="16" \
    --eval_strategy="steps" \
    --learning_rate="3e-4" \
    --warmup_steps="500" \
    --fp16 \
    --freeze_feature_extractor \
    --save_steps="400" \
    --eval_steps="400" \
    --save_total_limit="3" \
    --logging_steps="400" \
    --group_by_length \
    --feat_proj_dropout="0.0" \
    --layerdrop="0.1" \
    --gradient_checkpointing \
    --do_train --do_eval
