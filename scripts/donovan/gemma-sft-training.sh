#!/bin/bash

source ../miniconda3/bin/activate
conda activate consistency-llms

# SAVE_PATH=/raid/users/ryan_cheng/checkpoints/education/llama3-8b-kto-prompt
SAVE_PATH="/mmfs1/gscratch/scrubbed/donoclay/checkpoints/education/gemma_2b-sft"
# PRETRAIN="google/gemma-2b-it"
PRETRAIN="/mmfs1/gscratch/scrubbed/donoclay/models/models--google--gemma-2b-it/snapshots/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad"
# set hugging face cache dir to /mmfs1/gscratch/scrubbed/donoclay/models/
export HF_HUB_CACHE="/mmfs1/gscratch/scrubbed/donoclay/models/"
WANDB_KEY="21c85189447fd554dc8f87f737ebfa5c57748e63"
DATASET_PATH="/mmfs1/home/donoclay/socialrl/donoclay/consistency_LLMs/training_data/out"

# set cuda memory allocation to be expandable_segments:True with environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup deepspeed --include localhost:0,1 --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset json@$DATASET_PATH \
   --input_key in_text \
   --output_key out_text \
   --train_batch_size 256 \
   --micro_train_batch_size 8 \
   --max_samples 500000 \
   --pretrain $PRETRAIN \
   --save_path $SAVE_PATH \
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
   --use_wandb $WANDB_KEY > sft_chatting.out &
