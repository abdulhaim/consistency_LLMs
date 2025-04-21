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

# global dictionary for configuration

def get_text_after_colon(agent_name, text):
    """
    Extracts only the target agent's first utterance.
    If other agents speak in the same completion, they are removed.
    """
    pattern = rf"{agent_name}:\s*(.*?)(?:\n|[A-Z][a-z]+:|$)"  # matches up to next speaker or newline
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # fallback: remove all leading speaker tags and return first sentence
        text = re.sub(r'^([A-Za-z]+:\s*)+', '', text)
        return text.strip().split('\n')[0]  # return first line as fallback

def get_first_name(full_name):
    return full_name.strip().split()[0]
    
def generate_conversation(config, p1, p2, p1_name, p2_name, pturn=1):
    stats['P1'] = p1
    stats['P2'] = p2
    stats['pturn'] = pturn
    print(config['agent1_model'])
    round_num = 0
    prompts["agent1_role"] = get_first_name(p1_name) 
    prompts["agent2_role"] = get_first_name(p2_name)
    while round_num < config['convo_length_limit']:

        conversation = ("".join(stats["conversation"]) if len(stats["conversation"]) != 0 else "You are starting the conversation.\n")
        if pturn == 1:
            prompt = prompts["dialogue_prompt"].replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                               .replace("%LISTENER_ROLE%", prompts["agent2_role"]) \
                                               .replace("%SPEAKER_BACKSTORY%", p1) \
                                               .replace("%CONVERSATION%", conversation)
            prompt += prompts["reminder_prompt"]
            pturn = 2
            if config["verbose"]:
                print(prompt)
                print()

            if round_num!=0: 
                prompt+= "Continue the conversation."
            if round_num >=10 and round_num<=19:
                prompt+= "You have " + str((config['convo_length_limit']-round_num)//2) + " rounds left." + "Make sure to conclude the conversation as your near the end."

            elif round_num>19:
                prompt+= "This is your concluding line in the conversation."

            prompt+="%SPEAKER_ROLE%:"
            response = completion_create(config['agent1_model'], config, prompt)
            response = get_text_after_colon(prompts["agent1_role"], response)
            stats["conversation"].append(f"{prompts["agent1_role"]}: " + response + "\n")
        
        else:
            prompt = prompts["dialogue_prompt"].replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                                               .replace("%LISTENER_ROLE%", prompts["agent1_role"]) \
                                               .replace("%SPEAKER_BACKSTORY%", p2) \
                                               .replace("%CONVERSATION%", conversation)

            prompt += prompts["reminder_prompt"]
            pturn = 1    
            if config["verbose"]:
                print(prompt)
                print()

            if round_num!=0: 
                prompt+= "Continue the conversation."
            if round_num >=10 and round_num<=19:
                prompt+= "You have " + str((config['convo_length_limit']-round_num)//2) + " rounds left." + "Make sure to conclude the conversation as your near the end."
            elif round_num>19:
                prompt+= "This is your concluding line in the conversation."
            
            prompt+="%SPEAKER_ROLE%:"
            response = completion_create(config['agent2_model'], config, prompt)
            response = get_text_after_colon(prompts["agent2_role"], response)
            stats["conversation"].append(f"{prompts["agent2_role"]}: " + response + "\n")
        round_num += 1

    stats["rounds"] = round_num
    if config['verbose']:
        print(stats["conversation"])
    return stats.copy()

def reset_stats(config):
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
