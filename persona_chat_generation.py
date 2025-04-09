from utils import *
import os
import glob
import re
import json
import random
import time
import pickle
from absl import app, flags
from tqdm import tqdm
from datetime import datetime
import openai
from openai import OpenAI
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

def generate_conversation(p1, p2, pturn=1):
    stats['P1'] = p1
    stats['P2'] = p2
    stats['pturn'] = pturn

    round_num = 0
    while round_num < config['convo_length_limit']:
        conversation = ("".join(stats["conversation"]) if len(stats["conversation"]) != 0 else "[You are starting the conversation.]")
        if pturn == 1:
            prompt = prompts["dialogue_prompt"].replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                               .replace("%LISTENER_ROLE%", prompts["agent2_role"]) \
                                               .replace("%SPEAKER_BACKSTORY%", p1) \
                                               .replace("%CONVERSATION%", conversation)
            pturn = 2
            if config["verbose"]:
                print(prompt)
                print()
            stats["conversation"].append("P1: " + completion_create(config['agent1_model'], config, prompt) + "\n")
        else:
            prompt = prompts["dialogue_prompt"].replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                                               .replace("%LISTENER_ROLE%", prompts["agent1_role"]) \
                                               .replace("%SPEAKER_BACKSTORY%", p2) \
                                               .replace("%CONVERSATION%", conversation)
            pturn = 1    
            if config["verbose"]:
                print(prompt)
                print()
            stats["conversation"].append("P2: " + completion_create(config['agent2_model'], config, prompt) + "\n")
        round_num += 1

    stats["rounds"] = round_num
    if config['verbose']:
        print(stats["conversation"])
    return stats.copy()

def reset_stats():
    stats_template = {
        "task_name": config['task_name'],
        "P1": "",
        "P2": "",
        "conversation": [],
        "pturn": 0, # beginning person (1 or 2)
        "index": -1,
        "timestamp": "",
        "rounds": 0,
        'conversation_only': True
    }
    for key, value in stats_template.items():
        stats[key] = value

def main(argv):
    global prompts
    init()

    with open('data/anthology/personas_updated.json', 'r') as f:
        personas_updated = json.load(f)

    with open('config/persona_chat/prompts.json', 'r') as f:
        prompts = json.load(f)
    
    write_file = (
        f"data/anthology/exp/"
        f"{config['agent1_model']}_{config['seed']}.json"
    )
    
    config["task_name"] = "Anthology"


    index_offset = load_stats_file(write_file)
    if index_offset > config['iterations']:
    #     setup_vllm()
    # else:
        print('Info: Skipping file!')
    
    personas_paired = np.array(personas_updated).reshape(-1, 2)
    for iteration in tqdm(range(config['iterations']-index_offset)):
        if index_offset < config['iterations']:
            reset_stats()

            p1_dict, p2_dict = personas_paired[index_offset]
            pturn = index_offset % 2 + 1

            generate_conversation(p1_dict['persona'], p2_dict['persona'], pturn)
            stats['index'] = (index_offset if config['write'] else -1)
            stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            index_offset += 1
            if config['write']:
                write_stats(write_file)

if __name__ == '__main__':
    app.run(main)