#!/bin/bash

python consistency_eval.py --task=Education --filename=/nfs/kun2/users/ryan_cheng/consistency_LLMs/education/exp/human_eval_convs_ryan.json --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp --max_iter=100
python consistency_eval.py --task=Education --exp_folder=/nfs/kun2/users/ryan_cheng/consistency_LLMs/education/mistral_runs --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp7
