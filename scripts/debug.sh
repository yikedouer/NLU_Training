#!/usr/bin/env bash

CURRENT_DIR="./"
TASK_NAME=${1:-'comment_intention'}
wandb offline
wandb online
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export PRETRAIN_DIR=$CURRENT_DIR/pretrained_model/bert-base-chinese
# export PRETRAIN_DIR=$CURRENT_DIR/pretrained_model/chinese-macbert-base
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs

python3 comment_trainer.py \
    --wandb_desc=comment_intention_rdrop \
    --model_type=bert \
    --model_name_or_path=$PRETRAIN_DIR \
    --task_name=$TASK_NAME \
    --do_lower_case \
    --do_train \
    --do_rdrop \
    --warmup_proportion=0.01 \
    --evaluate_during_training \
    --data_dir=${DATA_DIR}/${TASK_NAME}/ \
    --train_max_seq_length=50 \
    --eval_max_seq_length=50 \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=200 \
    --learning_rate=3e-5 \
    --liner_learning_rate=2e-4 \
    --intent_loss_coef=1 \
    --kl_loss_intent_coef=1 \
    --intent_loss_type=multi_class_focal_loss \
    --num_train_epochs=10 \
    --logging_steps=-1 \
    --save_steps=1000000 \
    --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
    --overwrite_output_dir \
    --seed=42 
    # --checkpoint=checkpoint-615
# multi_class_focal_loss
# bs=32，lr=3e-5
