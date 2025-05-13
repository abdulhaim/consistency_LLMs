#!/bin/bash

python consistency_eval.py --task=Education --filename=/nfs/kun2/users/ryan_cheng/consistency_LLMs/education/exp/05.12.25/gemma-2-2b-it_0_852.json --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp
python consistency_eval.py --task=Education --filename=/nfs/kun2/users/ryan_cheng/consistency_LLMs/education/exp/05.12.25/Llama-3.1-8B-Instruct_0_138.json --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp
