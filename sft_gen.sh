#!/bin/bash

deepspeed --include localhost:4,5,6 --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset json@/nfs/kun2/users/ryan_cheng/consistency_LLMs/training_data/out \
   --input_key in_text \
   --output_key out_text \
   --train_batch_size 240 \
   --micro_train_batch_size 8 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
   --save_path /raid/users/ryan_cheng2/checkpoints/therapy/llama3-8b-sft-large-token \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --packing_samples \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --use_wandb 1e3fbbf6aeaa60fb339e7c43b375cb2be8aa7f5f

export CUDA_VISIBLE_DEVICES=1
python patient-therapist.py