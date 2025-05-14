#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import logging

os.environ.pop("HF_HUB_OFFLINE", None)
logging.getLogger().setLevel(logging.ERROR)  # or logging.CRITICAL

import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

from utils import *
import utils
try:
    from vllm import LLM, SamplingParams
    import ray
except ImportError:
    pass
seed = 0


import os

# In[3]:


from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Tianyi-Lab/Personas")
personas_chatting_array = ds['train']['Llama-3.1-70B-Instruct_descriptive_persona']

np.random.seed(seed)
random_indices = np.random.choice(len(personas_chatting_array), size=200, replace=False).astype(int)
personas_chatting_array = np.array(personas_chatting_array)[random_indices]


import json
import shutil
import os
config_chatting = {'agent1_role': 'Person A',
                  'agent2_role': 'Person B',
                  'agent1_prompt': "You are %SPEAKER_ROLE%, and you are having an online conversation with %LISTENER_ROLE%. Each of you is chatting to get to know each other, taking turns asking questions, and sharing stories about your lives, careers, and experiences. The goal is to try to find something in common. Do not only ask questions, you should also share details about yourself. This is a brief story that you have written about yourself: %SPEAKER_BACKSTORY%",
                  'agent2_prompt': "You are %SPEAKER_ROLE%, and you are having an online conversation with %LISTENER_ROLE%. Each of you is chatting to get to know each other, taking turns asking questions, and sharing stories about your lives, careers, and experiences. The goal is to try to find something in common. Do not only ask questions, you should also share details about yourself. This is a brief story that you have written about yourself: %SPEAKER_BACKSTORY%",
                  'reminder_prompt': "Keep your response very brief — 2 sentences or less. Do NOT repeat anything you've already said: \n",
                  "scenario": "Two people are getting to know one another through an online conversation. ",
                  'eval_prompts': {
                  'prompt_consistency': """You are evaluating whether the following utterance by %SPEAKER_ROLE% is consistent with their given background and the task instructions. Answer YES if the line directly contradicts any part of the background or task descriptio—this includes contradictions in facts, behavior, beliefs, emotional stance, or conversational strategy. Answer NO if the line introduces new details, but those details are **plausibly aligned** with the background. Be strict and literal in your evaluation: if the line violates the persona’s stated traits, mark it as a contradiction even if the violation is subtle. Background for %SPEAKER_ROLE%: %SPEAKER_BACKSTORY% Line spoken by %SPEAKER_ROLE%: %SPEAKER_LINE%  Answer YES if the line contradicts the background, and NO if it does not. Then, explain your answer in one sentence. Be precise and avoid vague justification."""}}

os.makedirs("chatting", exist_ok=True)
with open("chatting/config_chatting.json", "w", encoding="utf-8") as f:
    json.dump(config_chatting, f, indent=4)

config_llm = {'agent1_model': '/nfs/kun2/users/ryan_cheng/checkpoints_raid/Chatting/llama3-8b-kto-sft-prompt',
             'agent2_model': 'Llama-3.1-8B-Instruct',
             'eval_model': 'Llama-3.1-8B-Instruct',
             'iterations': 10,
             'verbose': False,
             'write': True,
             'convo_length_limit': 10,
             'max_tokens': 256,
             'gpus': 1,
             'seed': 0,
             'fp8': True,
             'task_name': 'Chatting',
             'model_dir': "/raid/users/ryan_cheng2/models/",
             'tmp_dir': "/raid/users/ryan_cheng2/tmp3/"}

# with open("chatting/Llama-3.1-8B-Instruct.json", "w", encoding="utf-8") as f:
#     json.dump(config_llm, f, indent=4)


# In[7]:


# def get_text_after_colon(label, text):
#     """
#     Extracts the text that appears after a given label followed by a colon.
#     Example: get_text_after_colon("Name", "Name: John Doe") returns "John Doe"
#     """
#     pattern = rf"{re.escape(label)}:\s*(.*)"
#     match = re.search(pattern, text)
#     return match.group(1).strip() if match else text

# personas_chatting = []

# for persona in personas_chatting_array:
#     name_prompt = "Given the following biography below, parse the name of the person. \nOnly output the name.\n" + persona + "\nName: "
#     name = completion_create("gpt-4o-mini", config_llm, name_prompt)
#     name = get_text_after_colon("Name", name)
#     personas_chatting.append({'persona': persona, 'name': name})


# In[8]:


# with open('chatting/config_chatting_personas.json', 'w') as f:
#     json.dump(personas_chatting, f, indent=4)


# In[9]:


with open("chatting/config_chatting_personas.json", "r") as f:
    personas_chatting = json.load(f)


# In[10]:


import re

def get_first_name(full_name):
    return full_name.strip().split()[0]
    
def clean_role_prefix(response, expected_role):
    """
    Removes repeated instances of the expected_role prefix at the start (e.g., 'Therapist: Therapist:'),
    and ensures the response begins with a single correct expected_role prefix.
    """
    pattern = rf"^(({re.escape(expected_role)}):\s*)+"
    cleaned = re.sub(pattern, '', response.strip(), flags=re.IGNORECASE)
    return cleaned
    
def is_role_confused(response, other_role):
    """
    Checks if the output starts with the wrong speaker tag.
    """
    if other_role + ":" in response:
        return True
    else: 
        return False

def generate_response(agent_model, expected_role, other_role, config_llm, prompt, max_retries=3):
    for _ in range(max_retries):
        response = completion_create(agent_model, config_llm, prompt)
        print(expected_role)
        if not is_role_confused(response, other_role):
            return clean_role_prefix(response, expected_role)
            
    return clean_role_prefix(response, expected_role)

def generate_chat(config_llm, p1, p2, p1_name, p2_name, pturn=1):
    stats['P1'] = p1
    stats['P2'] = p2
    stats['pturn'] = pturn
    config_chatting["agent1_role"] = get_first_name(p1_name)
    config_chatting["agent2_role"] = get_first_name(p2_name)

    round_num = 0
    while round_num < config_llm['convo_length_limit']:
        conversation = ("".join([turn[1] if isinstance(turn, tuple) else turn for turn in stats["conversation"]]) if len(stats["conversation"]) != 0 else "You are starting the conversation.\n")
        
        if pturn == 1:
            prompt = config_chatting["agent1_prompt"]
            pturn = 2
            if config_llm["verbose"]:
                print(prompt)
                print()

            if round_num!=0: 
                prompt+= "Your conversation so far is below:\nConversation: %CONVERSATION%"
                
            if round_num >=config_llm['convo_length_limit']*2-11 and round_num<=config_llm['convo_length_limit']*2-1:
                prompt+= "You have " + str((config_llm['convo_length_limit']-round_num)//2) + " rounds left." + "Make sure to conclude the conversation as you're near the end."

            elif round_num>config_llm['convo_length_limit']*2-1:
                prompt+= "This is your concluding line in the conversation."

            if round_num!=0: 
                prompt+= "Continue the conversation with " + config_chatting["agent2_role"] +  ". Remember you are " +  config_chatting["agent1_role"] + "."
                 
            prompt += config_chatting["reminder_prompt"] + "DO NOT PREFACE THE RESPONSE WITH THIRD-PERSON STATEMENTS SUCH AS \"Sure, here's a response from...\"\n"
            prompt+="%SPEAKER_ROLE%:"
            prompt = prompt.replace("%SPEAKER_ROLE%", config_chatting["agent1_role"]) \
                           .replace("%LISTENER_ROLE%", config_chatting["agent2_role"]) \
                           .replace("%SPEAKER_BACKSTORY%", p1) \
                           .replace("%CONVERSATION%", conversation)
            
            response = generate_response(config_llm['agent1_model'], config_chatting["agent1_role"], config_chatting["agent2_role"], config_llm, prompt)
            # while "\n\n" in response:
            #     print(response)
            #     response = generate_response(config_llm['agent1_model'], config_chatting["agent1_role"], config_chatting["agent2_role"], config_llm, prompt)
            stats["conversation"].append((round_num, f"{config_chatting["agent1_role"]}: " + response + "\n"))
        
        else:
            prompt = config_chatting["agent2_prompt"]
            pturn = 1    
            if config_llm["verbose"]:
                print(prompt)
                print()

            if round_num!=0: 
                prompt+= "Your conversation so far is below:\nConversation: %CONVERSATION%"
            if round_num >=config_llm['convo_length_limit']*2-11 and round_num<=config_llm['convo_length_limit']*2-1:
                prompt+= "You have " + str((config_llm['convo_length_limit']-round_num)//2) + " rounds left." + "Make sure to conclude the conversation as you're near the end."
            elif round_num>config_llm['convo_length_limit']*2-1:
                prompt+= "This is your concluding line in the conversation."

            if round_num!=0: 
                prompt+= "Continue the conversation with " + config_chatting["agent1_role"] +  ". Remember you are " +  config_chatting["agent2_role"] + "."

            prompt += config_chatting["reminder_prompt"] + "DO NOT PREFACE THE RESPONSE WITH THIRD-PERSON STATEMENTS SUCH AS \"Sure, here's a response from...\"\n"
            prompt+="%SPEAKER_ROLE%:"
            prompt = prompt.replace("%SPEAKER_ROLE%", config_chatting["agent2_role"]) \
                           .replace("%LISTENER_ROLE%", config_chatting["agent1_role"]) \
                           .replace("%SPEAKER_BACKSTORY", p2) \
                           .replace("%CONVERSATION%", conversation)

            response = generate_response(config_llm['agent2_model'], config_chatting["agent1_role"], config_chatting["agent2_role"], config_llm, prompt)
            
            # while "\n\n" in response:
            #     print(response)
            #     response = generate_response(config_llm['agent2_model'], config_chatting["agent2_role"], config_chatting["agent1_role"], config_llm, prompt)
            stats["conversation"].append((round_num, f"{config_chatting["agent2_role"]}: " + response + "\n"))
        round_num += 1

    stats["rounds"] = round_num
    if config_llm['verbose']:
        print(stats["conversation"])
    return stats.copy()

def reset_stats():
    stats_template = {
        "task_name": config_llm['task_name'],
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


# In[11]:


import os
import random
from datetime import datetime
import utils
utils.config = config_llm

current_date = str(datetime.now().strftime("%m.%d.%y"))
output_dir = f"chatting/exp/{current_date}"
os.makedirs(output_dir, exist_ok=True)

# Generate unique random number for filename
def generate_unique_file_number(output_dir, prefix, seed, extension=".json"):
    while True:
        rand_num = random.randint(0, 1000)
        filename = f"{prefix}_{seed}_{rand_num}{extension}"
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            return rand_num

unique_num = generate_unique_file_number(
    output_dir,
    config_llm['agent1_model'],
    config_llm['seed']
)

# File to write output to
# In[12]:


lengths = [10, 20, 40, 60]
write_files = [os.path.join(output_dir, f"sft_{config_llm['agent1_model'][config_llm['agent1_model'].rfind('/')+1:]}_{config_llm['seed']}_{length}_{unique_num}.json") for length in lengths]
if __name__ == "__main__":
    # index_offset = load_stats_file(write_file)
    for i in range(1):
        for p_dict1, p_dict2 in tqdm(np.array(personas_chatting)[:20].reshape(-1, 2)):
            for j, convo_length in enumerate(lengths):
                index_offset = load_stats_file(write_files[j])
                config_llm['convo_length_limit'] = convo_length
                reset_stats()
                conversation = generate_chat(
                    config_llm,
                    p_dict1["persona"], 
                    p_dict2["persona"],
                    p_dict1["name"], 
                    p_dict2["name"], 
                    pturn=1
                )
                stats['index'] = index_offset
                stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                write_stats(write_files[j])
                index_offset += 1
