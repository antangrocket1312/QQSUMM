#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

size=xl
seed=2
port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE=data/train/train.jsonl
EVAL_FILES=data/test/test.jsonl
PASSAGES=data/test/copora/input_reviews.jsonl
INDEX=data/test/copora/index/
SAVE_DIR=exps
EXPERIMENT_NAME=atlas-${size}-seed${seed}
NCTX=50

# INFERENCE
python inference.py \
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
    --model_path ${SAVE_DIR}/${EXPERIMENT_NAME}-lgret-lglm/checkpoint/step-34 \
    --model_name_or_path ./exps/${EXPERIMENT_NAME}-lgret-lglm/checkpoint_reader \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --rank_threshold 1.2 \
    --clustering_threshold 1.2 \
    --mean_cluster_similarity_to_query_threshold 1.15 \
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
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --data_path ./physionet.org/structured_CoT_brief_hospital_course_no_outlier_mistral.pkl \
    --bf16 False \
    --fp16 True \
    --output_dir ./brief_hospital_course_mistral_gptq_1_epoch_r128_64a_new_pertinent_2_new_form \
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