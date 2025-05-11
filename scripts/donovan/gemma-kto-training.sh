#!/bin/bash

# Example KTO training pipeline, perhaps easier to run each command in order rather than editing this file
# edit relevant variables

# conda activate consistency_
source ../miniconda3/bin/activate
conda activate consistency-llms

# modify task flag variable with one of [Chatting, Education, Therapy]
# other tasks currently still need to be supported in jsonl_gen.py
# takes in files in training_data/in and outputs training jsons in training_data/out
# python jsonl_gen.py --task=Chatting

# SAVE_PATH=/raid/users/ryan_cheng/checkpoints/education/llama3-8b-kto-prompt
SCRUBBED_DIR="/mmfs1/gscratch/scrubbed/donoclay"
SAVE_PATH="/mmfs1/gscratch/scrubbed/donoclay/checkpoints/education/gemma_2b-kto-sft-prompt-run3/"
# PRETRAIN="google/gemma-2b-it"
PRETRAIN="/mmfs1/gscratch/scrubbed/donoclay/models/models--google--gemma-2b-it/snapshots/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad"
# set hugging face cache dir to /mmfs1/gscratch/scrubbed/donoclay/models/
HF_HUB_CACHE="/mmfs1/gscratch/scrubbed/donoclay/models/"

WANDB_KEY="21c85189447fd554dc8f87f737ebfa5c57748e63"
DATASET_PATH="/mmfs1/home/donoclay/socialrl/donoclay/consistency_LLMs/training_data/out"

# set cuda memory allocation to be expandable_segments:True with environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# specify gpu numbers to host on after localhost:
  # around 15 gb
    # directory with train.jsonl and test.jsonl
    # wandb key to monitor run stats
nohup deepspeed --include localhost:0,1 --master_port 61000 --module openrlhf.cli.train_kto \
   --save_path $SAVE_PATH \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain $PRETRAIN \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --dataset json@$DATASET_PATH \
   --input_key in_text \
   --output_key out_text \
   --label_key score \
   --flash_attn \
   --beta 0.1 \
   --gradient_checkpointing \
   --use_wandb $WANDB_KEY > $SCRUBBED_DIR/logs/kto_education.out & 