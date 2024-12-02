#!/usr/bin/env bash

CURRENT_DIR="./"
TASK_NAME=${1:-'ind_car_intent'}
# wandb offline
wandb online
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export PRETRAIN_DIR=$CURRENT_DIR/pretrained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs

# global_pointer params
python3 intent_trainer.py \
    --wandb_desc=car2_rm_longtail_720data \
    --model_type=bert_intent \
    --model_name_or_path=$PRETRAIN_DIR \
    --task_name=$TASK_NAME \
    --do_lower_case \
    --do_train \
    --evaluate_during_training \
    --do_rdrop \
    --data_dir=$DATA_DIR/${TASK_NAME}/ \
    --train_max_seq_length=153 \
    --eval_max_seq_length=153 \
    --per_gpu_train_batch_size=100 \
    --per_gpu_eval_batch_size=800 \
    --learning_rate=4e-5 \
    --liner_learning_rate=2e-3 \
    --in_context \
    --multi_label \
    --intent_loss_coef=5 \
    --kl_loss_intent_coef=0.01 \
    --intent_loss_type=multi_label_circle_loss \
    --num_cycles 2 \
    --num_train_epochs=50 \
    --logging_steps=-1 \
    --save_steps=-1 \
    --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
    --overwrite_output_dir \
    --seed=42 

# softmax decoder params
# python3 trainer.py \
#     --wandb_desc=softmax_decoder \
#     --model_type=bert \
#     --model_name_or_path=$PRETRAIN_DIR \
#     --task_name=$TASK_NAME \
#     --do_lower_case \
#     --do_train \
#     --evaluate_during_training \
#     --do_rdrop \
#     --data_dir=$DATA_DIR/${TASK_NAME}/ \
#     --train_max_seq_length=153 \
#     --eval_max_seq_length=153 \
#     --per_gpu_train_batch_size=100 \
#     --per_gpu_eval_batch_size=800 \
#     --learning_rate=4e-5 \
#     --liner_learning_rate=2e-3 \
#     --slot_decoder_name=softmax \
#     --markup=bio \
#     --intent_loss_coef=10 \
#     --slot_loss_coef=2 \
#     --kl_loss_intent_coef=0.01 \
#     --kl_loss_slot_coef=0.01 \
#     --slot_loss_type=ce \
#     --intent_loss_type=multi_label_circle_loss \
#     --num_train_epochs=50 \
#     --logging_steps=-1 \
#     --save_steps=-1 \
#     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#     --overwrite_output_dir \
#     --seed=42 

# crf decoder params
# python3 trainer.py \
#     --wandb_desc=crf_eval_debug \
#     --model_type=bert \
#     --model_name_or_path=$PRETRAIN_DIR \
#     --task_name=$TASK_NAME \
#     --do_lower_case \
#     --do_train \
#     --evaluate_during_training \
#     --do_rdrop \
#     --data_dir=$DATA_DIR/${TASK_NAME}/ \
#     --train_max_seq_length=153 \
#     --eval_max_seq_length=153 \
#     --per_gpu_train_batch_size=100 \
#     --per_gpu_eval_batch_size=800 \
#     --learning_rate=4e-5 \
#     --liner_learning_rate=2e-3 \
#     --slot_decoder_name=crf \
#     --markup=bio \
#     --intent_loss_coef=10 \
#     --slot_loss_coef=1 \
#     --kl_loss_intent_coef=0.01 \
#     --kl_loss_slot_coef=0.01 \
#     --slot_loss_type=ce \
#     --intent_loss_type=multi_label_circle_loss \
#     --num_train_epochs=50 \
#     --logging_steps=-1 \
#     --save_steps=-1 \
#     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#     --overwrite_output_dir \
#     --seed=42 
