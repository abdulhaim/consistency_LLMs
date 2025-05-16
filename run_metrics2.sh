#!/bin/bash

python consistency_eval.py --task=Chatting --exp_folder=./chatting/exp/05.06.25 --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp
python consistency_eval.py --task=Chatting --exp_folder=/nfs/kun2/users/ryan_cheng/consistency_LLMs/chatting/exp/05.13.25 --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp234 --max_iter=25

