#!/usr/bin/env bash

CURRENT_DIR=`pwd`
# TASK_NAME="ind_car_nlu"
TASK_NAME=${1:-'ind_car_nlu'}

export PRETRAIN_DIR=$CURRENT_DIR/pretrained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
# wandb offline
# wandb online

python3 -m torch.distributed.launch --nproc_per_node=2 trainer.py \
    --wandb_desc=ddp_r_drop_0.05_intent5 \
    --model_type=bert \
    --model_name_or_path=$PRETRAIN_DIR \
    --task_name=$TASK_NAME \
    --do_lower_case \
    --do_train \
    --do_eval \
    --do_adv \
    --do_rdrop \
    --data_dir=$DATA_DIR/${TASK_NAME}/ \
    --train_max_seq_length=153 \
    --eval_max_seq_length=153 \
    --per_gpu_train_batch_size=100 \
    --per_gpu_eval_batch_size=200 \
    --learning_rate=4e-5 \
    --liner_learning_rate=2e-3 \
    --slot_decoder=softmax \
    --slot_loss_coef=1 \
    --slot_loss_type=ce \
    --intent_loss_type=multi_label_circle_loss \
    --num_train_epochs=5 \
    --logging_steps=-1 \
    --save_steps=-1 \
    --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
    --overwrite_output_dir \
    --overwrite_cache \
    --seed=42 

    # --checkpoint=checkpoint-4334