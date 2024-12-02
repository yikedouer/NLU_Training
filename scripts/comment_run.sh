#!/usr/bin/env bash

CURRENT_DIR="./"
TASK_NAME=${1:-'comment_intention'}
# wandb offline
wandb online
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export PRETRAIN_DIR=$CURRENT_DIR/pretrained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs

python3 intent_trainer.py \
    --wandb_desc=comment \
    --model_type=bert_intent \
    --model_name_or_path=$PRETRAIN_DIR \
    --task_name=$TASK_NAME \
    --do_lower_case \
    --do_train \
    --warmup_proportion=0.01 \
    --evaluate_during_training \
    --do_rdrop \
    --data_dir=$DATA_DIR/${TASK_NAME}/ \
    --train_max_seq_length=50 \
    --eval_max_seq_length=50 \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=400 \
    --learning_rate=3e-5 \
    --liner_learning_rate=2e-4 \
    --intent_loss_coef=1 \
    --kl_loss_intent_coef=1 \
    --intent_loss_type=multi_class_focal_loss \
    --num_cycles 1 \
    --num_train_epochs=15 \
    --logging_steps=-1 \
    --save_steps=1000000 \
    --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
    --overwrite_output_dir \
    --seed=42 
# backup 
# python3 intent_trainer.py \
#     --wandb_desc=backup_bash \
#     --model_type=bert_intent \
#     --model_name_or_path=$PRETRAIN_DIR \
#     --task_name=$TASK_NAME \
#     --do_lower_case \
#     --do_train \
#     --warmup_proportion=0.01 \
#     --evaluate_during_training \
#     --do_rdrop \
#     --data_dir=$DATA_DIR/${TASK_NAME}/ \
#     --train_max_seq_length=50 \
#     --eval_max_seq_length=50 \
#     --per_gpu_train_batch_size=32 \
#     --per_gpu_eval_batch_size=400 \
#     --learning_rate=3e-5 \
#     --liner_learning_rate=2e-4 \
#     --intent_loss_coef=1 \
#     --kl_loss_intent_coef=1 \
#     --intent_loss_type=multi_class_focal_loss \
#     --num_cycles 1 \
#     --num_train_epochs=15 \
#     --logging_steps=-1 \
#     --save_steps=1000000 \
#     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#     --overwrite_output_dir \
#     --seed=42 
# Awesome R-Drop 
# python3 intent_trainer.py \
#     --wandb_desc=comment \
#     --model_type=bert_intent \
#     --model_name_or_path=$PRETRAIN_DIR \
#     --task_name=$TASK_NAME \
#     --do_lower_case \
#     --do_train \
#     --warmup_proportion=0.01 \
#     --evaluate_during_training \
#     --do_rdrop \
#     --data_dir=$DATA_DIR/${TASK_NAME}/ \
#     --train_max_seq_length=50 \
#     --eval_max_seq_length=50 \
#     --per_gpu_train_batch_size=32 \
#     --per_gpu_eval_batch_size=400 \
#     --learning_rate=3e-5 \
#     --liner_learning_rate=2e-4 \
#     --intent_loss_coef=1 \
#     --kl_loss_intent_coef=4 \
#     --intent_loss_type=multi_class_focal_loss \
#     --num_cycles 1 \
#     --num_train_epochs=15 \
#     --logging_steps=-1 \
#     --save_steps=1000000 \
#     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#     --overwrite_output_dir \
#     --seed=42 


# python3 intent_trainer.py \
#     --wandb_desc=comment \
#     --model_type=bert_intent \
#     --model_name_or_path=$PRETRAIN_DIR \
#     --task_name=$TASK_NAME \
#     --do_lower_case \
#     --do_train \
#     --warmup_proportion=0.01 \
#     --evaluate_during_training \
#     --do_rdrop \
#     --data_dir=$DATA_DIR/${TASK_NAME}/ \
#     --train_max_seq_length=50 \
#     --eval_max_seq_length=50 \
#     --per_gpu_train_batch_size=32 \
#     --per_gpu_eval_batch_size=400 \
#     --learning_rate=3e-5 \
#     --liner_learning_rate=2e-4 \
#     --intent_loss_coef=1 \
#     --kl_loss_intent_coef=1 \
#     --intent_loss_type=multi_class_focal_loss \
#     --num_cycles 1 \
#     --num_train_epochs=15 \
#     --logging_steps=-1 \
#     --save_steps=1000000 \
#     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#     --overwrite_output_dir \
#     --seed=42 

# 08/15/2023 22:24:56 - INFO - root -   Intent Report for Step 1080: 
# 08/15/2023 22:24:56 - INFO - root -   intent    precision       recall  f1-score        support
# 08/15/2023 22:24:56 - INFO - root -   非高意向  0.958   0.965   0.961   1474
# 08/15/2023 22:24:56 - INFO - root -   高意向    0.861   0.833   0.847   378
# 08/15/2023 22:24:56 - INFO - root -   micro     0.909   0.899   0.904   1852
#高意向    0.853   0.847   0.85    378
# python3 intent_trainer.py \
#     --wandb_desc=comment \
#     --model_type=bert_intent \
#     --model_name_or_path=$PRETRAIN_DIR \
#     --task_name=$TASK_NAME \
#     --do_lower_case \
#     --do_train \
#     --warmup_proportion=0.01 \
#     --evaluate_during_training \
#     --do_rdrop \
#     --data_dir=$DATA_DIR/${TASK_NAME}/ \
#     --train_max_seq_length=50 \
#     --eval_max_seq_length=50 \
#     --per_gpu_train_batch_size=32 \
#     --per_gpu_eval_batch_size=400 \
#     --learning_rate=1e-5 \
#     --liner_learning_rate=2e-4 \
#     --intent_loss_coef=1 \
#     --kl_loss_intent_coef=1 \
#     --intent_loss_type=multi_class_focal_loss \
#     --num_cycles 1 \
#     --num_train_epochs=10 \
#     --logging_steps=-1 \
#     --save_steps=1000000 \
#     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#     --overwrite_output_dir \
#     --seed=42 

# best params
# python3 intent_trainer.py \
#     --wandb_desc=comment \
#     --model_type=bert_intent \
#     --model_name_or_path=$PRETRAIN_DIR \
#     --task_name=$TASK_NAME \
#     --do_lower_case \
#     --do_train \
#     --warmup_proportion=0.01 \
#     --evaluate_during_training \
#     --do_rdrop \
#     --data_dir=$DATA_DIR/${TASK_NAME}/ \
#     --train_max_seq_length=50 \
#     --eval_max_seq_length=50 \
#     --per_gpu_train_batch_size=32 \
#     --per_gpu_eval_batch_size=400 \
#     --learning_rate=3e-5 \
#     --liner_learning_rate=2e-4 \
#     --intent_loss_coef=1 \
#     --kl_loss_intent_coef=1 \
#     --intent_loss_type=multi_class_focal_loss \
#     --num_cycles 1 \
#     --num_train_epochs=10 \
#     --logging_steps=-1 \
#     --save_steps=1000000 \
#     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#     --overwrite_output_dir \
#     --seed=42 
