#!/bin/bash

python consistency_eval.py --task=Therapy --exp_folder=./therapy/exp/05.08.25 --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp4 --max_iter=100


python consistency_eval.py --task=Therapy --filename=/nfs/kun2/users/ryan_cheng/consistency_LLMs/therapy/exp/human_eval_convs.json --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp4 --max_iter=100
python consistency_eval.py --task=Therapy --filename=/nfs/kun2/users/ryan_cheng/consistency_LLMs/therapy/exp/05.15.25/ppo_sft_new_lr_Llama-3.1-8B-Instruct_0_433.json --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp4 --max_iter=100