#!/bin/bash

python consistency_eval.py --task=Education --exp_folder=./data/education/exp/04.30.25 --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp
python consistency_eval.py --task=Education --exp_folder=./education/exp/05.06.25 --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp
python consistency_eval.py --task=Therapy --exp_folder=./therapy/exp/05.08.25 --gpus=2 --model_dir=/raid/users/ryan_cheng2/models --tmp_dir=/raid/users/ryan_cheng2/tmp
