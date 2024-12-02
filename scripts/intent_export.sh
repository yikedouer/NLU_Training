#!/usr/bin/env bash

CURRENT_DIR="./"
# TASK_NAME="ind_car_nlu"
TASK_NAME=${1:-'ind_car_v2_nlu'}
# USE_DDP=${1:-'ind_car_nlu'}
wandb offline
# wandb online
export PYTORCH_NVFUSER_DISABLE=fallback
export PYTORCH_JIT_LOG_LEVEL=manager.cpp
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export PRETRAIN_DIR=$CURRENT_DIR/pretrained_model/bert-base-chinese
# export PRETRAIN_DIR=$CURRENT_DIR/pretrained_model/chinese-macbert-base
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs

python3 intent_trainer.py \
    --wandb_desc=rdrop_slot_weight1_no_ema_macbert \
    --model_type=bert_intent \
    --model_name_or_path=$PRETRAIN_DIR \
    --task_name=$TASK_NAME \
    --do_lower_case \
    --do_export \
    --do_rdrop \
    --in_context \
    --data_dir=$DATA_DIR/${TASK_NAME}/ \
    --train_max_seq_length=153 \
    --eval_max_seq_length=153 \
    --slot_decoder=global_pointer \
    --markup=span \
    --multi_label \
    --intent_loss_coef=5 \
    --slot_loss_coef=2 \
    --kl_loss_intent_coef=0.01 \
    --kl_loss_slot_coef=0.01 \
    --slot_loss_type=multi_label_circle_loss \
    --intent_loss_type=multi_label_circle_loss \
    --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
    --overwrite_output_dir \
    --seed=42 \
    --checkpoint=checkpoint-4481
