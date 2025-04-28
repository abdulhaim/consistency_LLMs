#!/usr/bin/env python

# Imports 
import os
import logging
import subprocess
import torch
import glob
import re
import json
import shutil
import time
import pickle
import random

from absl import app, flags
from tqdm import tqdm
from datetime import datetime
import openai
from openai import OpenAI
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
try:
    from vllm import LLM, SamplingParams
    import ray
except ImportError:
    pass
    
# File Imports 
from utils import *
import utils
    
seed = 0

# Settings
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
logging.getLogger().setLevel(logging.ERROR)  # or logging.CRITICAL
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

## Utils 
def get_freest_cuda_device():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, encoding='utf-8')
    memory_free = [int(x) for x in result.stdout.strip().split('\n')]
    return memory_free.index(max(memory_free))


def setup_():
    best_gpu = get_freest_cuda_device()
    device = torch.device(f"cuda:{best_gpu}")
    print(f"Using GPU: {device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

    with open(os.path.abspath('../openai_key'), 'r') as f:
        utils.client = OpenAI(api_key=f.read().rstrip('\n'))

    with open("therapy/Llama-3.1-8B-Instruct.json", "w", encoding="utf-8") as f:
        json.dump(config_llm, f, indent=4)
    
    with open("therapy/config_therapy_personas.json", "r", encoding="utf-8") as f:
        personas_therapy = json.load(f)


def create_therapy_personas()
    with open("../token.txt", "r") as f:
        token = f.read().strip()

    from huggingface_hub import login
    login(token=token)

    
    from datasets import load_dataset
    ds = load_dataset("ShenLab/MentalChat16K")
    train_data = ds['train']

    # Collect all personas with > 200 words into a dictionary
    persona_dict = {}
    count = 0
    for persona_sample in train_data:
        patient_persona = persona_sample['input']
        if count_words(patient_persona) > 200:
            persona_dict[f"persona_{count}"] = patient_persona
            count += 1
    
    # Randomly sample 100 keys
    sampled_keys = random.sample(sorted(persona_dict.keys()), 100)
    
    # Create a new dict with only the sampled entries
    sampled_persona_dict = {k: persona_dict[k] for k in sampled_keys}
    
    # Save to JSON
    with open("therapy/MentalChat16K_sampled_personas.json", "w") as f:
        json.dump(sampled_persona_dict, f, indent=2)

def count_words(text):
    if text!=None:
        words = text.split()
        return len(words)
    else:
        return 0

# Generate unique random number for filename
def generate_unique_file_number(output_dir, prefix, seed, extension=".json"):
    while True:
        rand_num = random.randint(0, 1000)
        filename = f"{prefix}_{seed}_{rand_num}{extension}"
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            return rand_num


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


def generate_conversation(config_llm, p1, p2, p1_name, p2_name, pturn=1):
    stats['P1'] = p1
    stats['P2'] = p2
    stats['pturn'] = pturn
    round_num = 0
    while round_num < config_llm['convo_length_limit']:
        conversation = ("".join([turn[1] if isinstance(turn, tuple) else turn for turn in stats["conversation"]]) if len(stats["conversation"]) != 0 else "You are starting the conversation.\n")

        if pturn == 1:
            prompt = config_therapy["agent1_prompt"]
            pturn = 2
            if config_llm["verbose"]:
                print(prompt)
                print()

            if round_num!=0: 
                prompt+= "Your conversation with the therapist so far is below:\nConversation: %CONVERSATION%"

            if round_num >=config_llm['convo_length_limit']*2-11 and round_num<=config_llm['convo_length_limit']*2-1:
                prompt+= "You have " + str((config_llm['convo_length_limit']-round_num)//2) + " rounds left." + "Make sure to conclude the conversation as your near the end."

            elif round_num>config_llm['convo_length_limit']*2-1:
                prompt+= "This is your concluding line in the conversation."

            if round_num!=0: 
                prompt+= "Continue the conversation with the therapist. Remember you are the patient."

            prompt += config_therapy["reminder_prompt"]
            prompt = prompt.replace("%SPEAKER_ROLE%", config_therapy["agent1_role"]) \
                           .replace("%LISTENER_ROLE%", config_therapy["agent2_role"]) \
                           .replace("%SPEAKER_BACKSTORY%", p1) \
                           .replace("%CONVERSATION%", conversation)

            prompt+="%SPEAKER_ROLE%:"
            response = generate_response(config_llm['agent1_model'], config_therapy["agent1_role"], config_therapy["agent2_role"], config_llm, prompt)
            stats["conversation"].append((round_num, f"{config_therapy["agent1_role"]}: " + response + "\n"))

        else:
            prompt = config_therapy["agent2_prompt"]
            pturn = 1    
            if config_llm["verbose"]:
                print(prompt)
                print()

            if round_num!=0: 
                prompt+= "Your conversation with the patient so far is below:\nConversation: %CONVERSATION%"
            if round_num >=config_llm['convo_length_limit']*2-11 and round_num<=config_llm['convo_length_limit']*2-1:
                prompt+= "You have " + str((config_llm['convo_length_limit']-round_num)//2) + " rounds left." + "Make sure to conclude the conversation as your near the end."
            elif round_num>config_llm['convo_length_limit']*2-1:
                prompt+= "This is your concluding line in the conversation."

            if round_num!=0: 
                prompt+= "Continue the conversation with the patient. Remember you are the therapist."

            prompt += config_therapy["reminder_prompt"]
            prompt = prompt.replace("%SPEAKER_ROLE%", config_therapy["agent2_role"]) \
                           .replace("%LISTENER_ROLE%", config_therapy["agent1_role"]) \
                           .replace("%CONVERSATION%", conversation)

            prompt+="%SPEAKER_ROLE%:"
            response = generate_response(config_llm['agent2_model'], config_therapy["agent2_role"], config_therapy["agent1_role"], config_llm, prompt)
            stats["conversation"].append((round_num, f"{config_therapy["agent2_role"]}: " + response + "\n"))
        round_num += 1

    stats["rounds"] = round_num
    if config_llm['verbose']:
        print(stats["conversation"])
    return stats.copy()

def generate_conversations():
    index_offset = load_stats_file(write_file)
    conversations = []    
    lengths = [10, 20, 40, 60]
    for i in range(1):
        for patient_dict in personas_therapy:
            for convo_length in lengths:
                config_llm['convo_length_limit'] = convo_length
                reset_stats()
                conversation = generate_conversation(
                    config_llm,
                    patient_dict["description"] + "" + patient_dict["strategy"], 
                    "",
                    "Patient", 
                    "Therapist", 
                    pturn=1
                )
                print(conversation)
                conversations.append(conversation)
                stats['index'] = index_offset
                stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                write_stats(write_file)
                index_offset += 1

    current_date = str(datetime.now().strftime("%m.%d.%y"))
    output_dir = f"therapy/exp/{current_date}"
    os.makedirs(output_dir, exist_ok=True)
    
    unique_num = generate_unique_file_number(
        output_dir,
        config_llm['agent1_model'],
        config_llm['seed']
    )
    
    # File to write output to
    write_file = os.path.join(output_dir, f"{config_llm['agent1_model']}_{config_llm['seed']}_{unique_num}.json")

if __name__ == "__main__":

    
    config = config_llm
    prompts = config_therapy
    for conversation in tqdm(conversations):
        conversation = eval_prompt_consistency(conversation, both_agents=False)
    
    write_stats(write_file)
    








eval_prompts = {
    "strategy_consistency": "%SCENARIO_DESC% Evaluate whether the following line spoken by %SPEAKER_ROLE% contradicts their background. First, describe the interpreted intention of the statement, and whether or not it aligns with the given background of %SPEAKER_ROLE%. Then, provide 1 sentence explaining your reasoning based on the strategy, followed by YES if the line contradicts the given background of %SPEAKER_ROLE%, and NO if it is consistent with the provided background of %SPEAKER_ROLE%. %SPEAKER_ROLE%'s strategy is described as follows:\n%SPEAKER_STRATEGY%\n %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n Provide 1 sentence of reasoning followed by YES or NO.\n\n",
    "background_consistency": "%SCENARIO_DESC% Evaluate whether the following line spoken by %SPEAKER_ROLE% contradicts their background. Provide 1 sentence explaining your reasoning based on the background, followed by YES if the line contradicts the given background of %SPEAKER_ROLE%, and NO if it is consistent with the provided background of %SPEAKER_ROLE%. %SPEAKER_ROLE%'s background is described as follows:\n%SPEAKER_BACKSTORY%\n %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n Provide 1 sentence of reason reasoning followed by YES or NO.\n\n",
    "combined_prompt_consistency": "%SCENARIO_DESC% Evaluate the intention behind the following line spoken by %SPEAKER_ROLE% and determine whether it contradicts their background. First, describe the interpreted intention of the statement, and whether or not it aligns with the given background of %SPEAKER_ROLE%. Then, answer YES if the line contradicts the given background of %SPEAKER_ROLE% or the intention does not align with the provided background, and answer NO if it does align with the provided background or the intention aligns with the background of %SPEAKER_ROLE%. %SPEAKER_ROLE%'s background is described as follows:\n%SPEAKER_BACKSTORY%\n %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n Provide your answer as 1 sentence explaining your reasoning based on the background and the interpreted intention, followed by YES or NO.\n\n",

    "pairwise_consistency":"%SCENARIO_DESC% For the following line spoken by %SPEAKER_ROLE%, answer YES if the line directly contradicts the provided line spoken by %LISTENER_ROLE%, and answer NO if the line does not contradict the provided line spoken by %LISTENER_ROLE%. %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n %LISTENER_ROLE% spoke the following line: \n%LISTENER_LINE%\n\n Answer YES if the line spoken by %SPEAKER_ROLE% contradicts the provided line spoken by %LISTENER_ROLE%, and answer NO if the line does not contradict the provided line spoken by %LISTENER_ROLE%, followed by 1 sentence of reasoning.\n\n",

    "backstory_test": "Based on the following background, generate a new fact-based multiple choice question with 5 choices addressed directly IN SECOND PERSON, along with its correct answer. Preface the question with 'Question:' and the answer with 'Answer:'.\n%SPEAKER_BACKSTORY%\n%PREVIOUS_QUESTIONS%",
    "answer_backstory": "You are %SPEAKER_ROLE%, and you are having a conversation with %LISTENER_ROLE%. Your background is:\n%SPEAKER_BACKSTORY%\n So far, the conversation is as below:\n%CONVERSATION%\n\n Based on your conversation above so far, answer the following multiple choice question.\n%BACKSTORY_QUESTION%\n",
    "grade_backstory": "As part of grading a test, determine whether the given answer %GIVEN_ANSWER% matches the following correct answer. Respond with either YES or NO.\nCorrect Answer: %CORRECT_ANSWER%\n"
}



with open("therapy/exp/04.24.25/Llama-3.1-8B-Instruct_0_250_consistency1.json", "w", encoding="utf-8") as f:
    json.dump(conversations, f, indent=4)




