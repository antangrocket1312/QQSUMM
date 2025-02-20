#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

size=xl
seed=2
port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE=data/train/train.jsonl
PRETRAINED_MODEL=checkpoints/models/atlas/${size}
PASSAGES=data/train/copora/input_reviews.jsonl
INDEX=data/train/copora/index/
SAVE_DIR=exps
FC_FILE=data/train/copora/gold_retrieved_comments.json
CLUSTER_FILE=data/train/copora/gold_comment_clusters.json
EXPERIMENT_NAME=atlas-${size}-seed${seed}

# WARMUP THE LM
python train.py \
    --shuffle \
    --fc_only \
    --fc_file ${FC_FILE} \
    --cluster_file ${CLUSTER_FILE} \
    --use_gradient_checkpoint_reader \
    --name ${EXPERIMENT_NAME} \
    --precision fp16 \
    --shard_optim --shard_grads \
    --reader_model_type mistralai/Mistral-7B-Instruct-v0.2 \
    --model_path ${PRETRAINED_MODEL} \
    --train_data ${TRAIN_FILE} \
    --per_gpu_batch_size 1 \
    --n_context ${NCTX} --retriever_n_context ${NCTX} \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port ${port} \
    --passages ${PASSAGES}\
    --save_index_path ${INDEX} \
    --seed ${seed} \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --bf16 False \
    --fp16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-4 \
    --weight_decay 0 \
    --warmup_ratio 0 \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --model_max_length 2048 \
    --q_lora True \
    --gradient_checkpointing True \
    --report_to "none" \
    --seed 42 \
    --optim "adamw_torch" \
    --ddp_find_unused_parameters False

# MAIN TRAINING
python train.py \
    --shuffle \
    --train_retriever \
    --use_gradient_checkpoint_retriever \
    --gold_score_mode ppmean \
    --use_gradient_checkpoint_reader \
    --lgret \
    --name ${EXPERIMENT_NAME}-lgret-lglm \
    --precision fp16 \
    --shard_optim --shard_grads \
    --reader_model_type mistralai/Mistral-7B-Instruct-v0.2 \
    --model_path ${SAVE_DIR}/${EXPERIMENT_NAME}/checkpoint/step-34 \
    --train_data ${TRAIN_FILE} \
    --fc_file ${FC_FILE} \
    --cluster_file ${CLUSTER_FILE} \
    --per_gpu_batch_size 1 \
    --n_context ${NCTX} --retriever_n_context ${NCTX} \
    --checkpoint_dir ${SAVE_DIR} \
    --retrieve_with_rerank \
    --eval_freq 30 \
    --main_port ${port} \
    --write_results \
    --passages ${PASSAGES}\
    --load_index_path ${INDEX} \
    --seed ${seed} \
    --model_name_or_path ./exps/atlas-xl-seed2/checkpoint_reader \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --bf16 False \
    --fp16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-4 \
    --weight_decay 0 \
    --warmup_ratio 0 \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --model_max_length 2048 \
    --q_lora True \
    --gradient_checkpointing True \
    --report_to "none" \
    --seed 42 \
    --optim "adamw_torch" \
    --ddp_find_unused_parameters False
