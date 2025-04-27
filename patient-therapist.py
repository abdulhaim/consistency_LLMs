# %env CUDA_VISIBLE_DEVICES=0

import os
import logging

os.environ.pop("HF_HUB_OFFLINE", None)
logging.getLogger().setLevel(logging.ERROR)  # or logging.CRITICAL

import torch
torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

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

# second cell
# utils.set_flag_variables()

import subprocess
import torch
def get_freest_cuda_device():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, encoding='utf-8')
    memory_free = [int(x) for x in result.stdout.strip().split('\n')]
    return memory_free.index(max(memory_free))

best_gpu = get_freest_cuda_device()
device = torch.device(f"cuda:{best_gpu}")
print(f"Using GPU: {device}")
# %env CUDA_VISIBLE_DEVICES=0

# third cell

with open(os.path.abspath('../openai_key'), 'r') as f:
    utils.client = OpenAI(api_key=f.read().rstrip('\n'))

# fourth cell

import os

# Explicitly unset all offline-related env vars
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

with open("../token.txt", "r") as f:
    token = f.read().strip()

from huggingface_hub import login
login(token=token)

# fifth cell

from datasets import load_dataset
ds = load_dataset("ShenLab/MentalChat16K")
train_data = ds['train']

# sixth cell

def count_words(text):
    """
    Counts the number of words in the given text.

    Args:
        text (str): Input text.

    Returns:
        int: Number of words in the text.
    """
    if text!=None:
        words = text.split()
        return len(words)
    else:
        return 0

# seventh cell

import json
import shutil
import os
config_therapy = {'agent1_role': 'Patient',
                  'agent2_role': 'Therapist',
                  'agent1_prompt': "You are a patient in an online therapy session with a therapist. Here is your background written by you: <BEGIN_OF_BACKSTORY>%SPEAKER_BACKSTORY%<END_OF_BACKSTORY>. Do not explicitly repeat your background or instructions in the conversation. Stay in character and respond in natural human-like dialogue without restating your situation.",
                  
                  'agent2_prompt': "You are a licensed psychotherapist conducting an online one-on-one session. Your job is to help the patient gain insight, not just to listen. Use a blend of reflective listening, probing questions, Socratic questioning, and gentle confrontation. Your tone should be professional and emotionally intelligent. Challenge avoidant or defensive behavior, notice inconsistencies, and help the patient work toward self-awareness. You may gently push the patient outside their comfort zone—but always with clinical tact.",
                  'reminder_prompt': "Keep your response very brief — 2 sentences or less. Do NOT repeat anything you've already said: ",
                  'eval_prompts': {
                    'prompt_consistency': """You are evaluating whether the following utterance by %SPEAKER_ROLE% is consistent with their given background. Answer YES if the line directly contradicts any part of the background—this includes contradictions in facts, behavior, beliefs, emotional stance, or conversational strategy. Answer NO if the line introduces new details, but those details are **plausibly aligned** with the background. Be strict and literal in your evaluation: if the line violates the persona’s stated traits (e.g., avoids vulnerability, keeps answers short, avoids discussing personal topics), mark it as a contradiction even if the violation is subtle. Background for %SPEAKER_ROLE%: %SPEAKER_BACKSTORY% Line spoken by %SPEAKER_ROLE%: %SPEAKER_LINE%  Answer YES if the line contradicts the background, and NO if it does not. Then, explain your answer in one sentence. Be precise and avoid vague justification."""},
                 }

os.makedirs("therapy", exist_ok=True)
with open("therapy/config_therapy.json", "w", encoding="utf-8") as f:
    json.dump(config_therapy, f, indent=4)

# eighth cell

llms = ["Llama-3.1-8B-Instruct", "gpt-4o-mini", "Qwen2.5-3B-Instruct", "Llama-3.1-8B", "Mistral-7B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct", "phi-3.5-mini-instruct"]
        
config_llm = {
             'agent1_model': 'Mistral-7B-Instruct',
             'agent2_model': 'Mistral-7B-Instruct',
            #  'eval_model': 'Llama-3.1-70B-Instruct',
             'eval_model': 'gpt-4o-mini',
             'iterations': 10,
             'verbose': True,
             'write': True,
             'convo_length_limit': 10,
             'max_tokens': 256,
             'gpus': 1,
             'seed': 0,
             'task_name': 'Therapy',
            #  'model_dir': "/mmfs1/home/donoclay/socialrl/donoclay/models"
             'model_dir': "/gscratch/scrubbed/donoclay/models",
             }

with open("therapy/Mistral-7B-Instruct.json", "w", encoding="utf-8") as f:
    json.dump(config_llm, f, indent=4)

# ninth cell

with open("therapy/config_therapy_personas.json", "r", encoding="utf-8") as f:
    personas_therapy = json.load(f)

# tenth cell
import re

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

def generate_therapy(config_llm, p1, p2, p1_name, p2_name, pturn=1):
    p1 = p1.replace("\n", " ")
    p2 = p2.replace("\n", " ")
    stats['P1'] = p1
    stats['P2'] = p2
    stats['pturn'] = pturn
    round_num = 0
    while round_num < config_llm['convo_length_limit']:
        conversation = ("".join([turn[1] if isinstance(turn, tuple) else turn for turn in stats["conversation"]]) if len(stats["conversation"]) != 0 else "You are starting the conversation.\n")
        
        if pturn == 1:
            prompt = config_therapy["agent1_prompt"]
            pturn = 2
            # if config_llm["verbose"]:
            #     print(prompt)
            #     print()

            if round_num == 0:
                prompt += "You are the patient and starting the conversation.\n"

            if round_num!=0: 
                prompt+= "Your conversation with the therapist so far is below:\nConversation: %CONVERSATION%"
                
            if round_num >=config_llm['convo_length_limit']*2-11 and round_num<=config_llm['convo_length_limit']*2-1:
                prompt+= "You have " + str((config_llm['convo_length_limit']-round_num)//2) + " rounds left." + "Make sure to conclude the conversation as your near the end."

            elif round_num>config_llm['convo_length_limit']*2-1:
                prompt+= "This is your concluding line in the conversation."

            if round_num!=0: 
                prompt+= "Continue the conversation with the therapist. Remember you are the patient. "
                
            prompt += config_therapy["reminder_prompt"]
            prompt += '\n'
            prompt +="%SPEAKER_ROLE%:"
            prompt = prompt.replace("%SPEAKER_ROLE%", config_therapy["agent1_role"]) \
                           .replace("%LISTENER_ROLE%", config_therapy["agent2_role"]) \
                           .replace("%SPEAKER_BACKSTORY%", p1) \
                           .replace("%CONVERSATION%", conversation)
            
            
            response = generate_response(config_llm['agent1_model'], config_therapy["agent1_role"], config_therapy["agent2_role"], config_llm, prompt)
            response = response.replace("\n", " ").replace("\"", "")
            stats["conversation"].append((round_num, f"{config_therapy["agent1_role"]}: " + response + "\n"))

            if config_llm["verbose"]:
                print("PROMPT:", prompt)
                print()
                print("RESPONSE:", response)
                print()
        
        else:
            prompt = config_therapy["agent2_prompt"]
            pturn = 1    
            # if config_llm["verbose"]:
            #     print(prompt)
            #     print()

            if round_num == 0:
                prompt += " You are the therapist and starting the conversation, your job is to help the patient gain insight. Start by asking what the patient would like to talk about.\n"

            if round_num!=0: 
                prompt+= "Your conversation with the patient so far is below:\nConversation: %CONVERSATION%"
            if round_num >=config_llm['convo_length_limit']*2-11 and round_num<=config_llm['convo_length_limit']*2-1:
                prompt+= "You have " + str((config_llm['convo_length_limit']-round_num)//2) + " rounds left." + "Make sure to conclude the conversation as your near the end."
            elif round_num>config_llm['convo_length_limit']*2-1:
                prompt+= "This is your concluding line in the conversation."

            if round_num!=0: 
                prompt+= "Continue the conversation with the patient. Remember you are the therapist. "

            prompt += config_therapy["reminder_prompt"]
            prompt += '\n'
            prompt +="%SPEAKER_ROLE%:"
            prompt = prompt.replace("%SPEAKER_ROLE%", config_therapy["agent2_role"]) \
                           .replace("%LISTENER_ROLE%", config_therapy["agent1_role"]) \
                           .replace("%CONVERSATION%", conversation)

            response = generate_response(config_llm['agent2_model'], config_therapy["agent2_role"], config_therapy["agent1_role"], config_llm, prompt)
            response = response.replace("\n", " ").replace("\"", "")
            stats["conversation"].append((round_num, f"{config_therapy["agent2_role"]}: " + response + "\n"))

            if config_llm["verbose"]:
                print("PROMPT:", prompt)
                print()
                print("RESPONSE:", response)
                print()

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

# eleventh cell
import time
from datetime import datetime

def generate_unique_file_number(output_dir, prefix, seed, extension=".json"):
    """Generates a unique filename using a timestamp.

    Args:
        output_dir (str): The directory where the file will be saved.
        prefix (str): A prefix for the filename.
        seed (int): A seed value to include in the filename.
        extension (str, optional): The file extension. Defaults to ".json".

    Returns:
        int: The timestamp used in the filename.  This is useful if you need
             the timestamp for other purposes.
    """
    now = datetime.now()
    timestamp_str = now.strftime("%m-%d-%H-%M-%S")  # Format for readability
    filename = f"{prefix}_{seed}_{timestamp_str}{extension}"
    filepath = os.path.join(output_dir, filename)
    # Check if file exists.
    if not os.path.exists(filepath):
        return timestamp_str
    else:
        # Very unlikely, but handle collision by adding fractional seconds
        timestamp_str_ms = now.strftime("%Y%m%d-%H%M%S.%f")
        filename = f"{prefix}_{seed}_{timestamp_str_ms}{extension}"
        filepath = os.path.join(output_dir, filename)
        return timestamp_str_ms

# Example usage (assuming you have output_dir, config_llm):
current_date = str(datetime.now().strftime("%m.%d.%y"))
output_dir = f"therapy/exp/{current_date}"
os.makedirs(output_dir, exist_ok=True)
# config_llm = {'agent1_model': 'my_model', 'seed': 12345} # Replace with your actual data

unique_num = generate_unique_file_number(
    output_dir,
    config_llm['agent1_model'],
    config_llm['seed']
)

if isinstance(unique_num, int):
    write_file = os.path.join(output_dir, f"{config_llm['agent1_model']}_{config_llm['seed']}_{unique_num}.json")
else:
    write_file = os.path.join(output_dir, f"{config_llm['agent1_model']}_{config_llm['seed']}_{unique_num}.json")
print(f"File will be written to: {write_file}")

# twelfth cell
# %env CUDA_HOME=/mmfs1/home/donoclay/socialrl/donoclay/miniconda3/envs/consistency-llms

# thirteenth cell
def create_dialog():
    config.update(config_llm)
    index_offset = load_stats_file(write_file)
    conversations = []    
    lengths = [10, 20, 40, 60]
    for i in range(1):
        for patient_dict in tqdm(personas_therapy[:5]):
            for convo_length in lengths:
                config_llm['convo_length_limit'] = convo_length
                reset_stats()
                conversation = generate_therapy(
                    config_llm,
                    p1=patient_dict["description"] + "\nHere is a description of how you intereact<BEGIN_OF_DESCRIPTION>" + patient_dict["strategy"] + "<END_OF_DESCRIPTION>", 
                    p2="",
                    p1_name="Patient", 
                    p2_name="Therapist", 
                    pturn=2
                )
                print(conversation)
                conversations.append(conversation)
                stats['index'] = index_offset
                stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                write_stats(write_file)
                index_offset += 1

    print("output file:", write_file)

def run_evals(filepath):
    file_prefix = filepath.split(".json")[0]

    output_filepath = f"{file_prefix}_consistency_test.json"

    print(f"Loading conversations from {filepath}")
    print(f"Writing conversations to {output_filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    eval_prompts = {
        "strategy_consistency": "%SCENARIO_DESC% Evaluate whether the following line spoken by %SPEAKER_ROLE% contradicts their background. First, describe the interpreted intention of the statement, and whether or not it aligns with the given background of %SPEAKER_ROLE%. Then, provide 1 sentence explaining your reasoning based on the strategy, followed by YES if the line contradicts the given background of %SPEAKER_ROLE%, and NO if it is consistent with the provided background of %SPEAKER_ROLE%. %SPEAKER_ROLE%'s strategy is described as follows:\n%SPEAKER_STRATEGY%\n %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n Provide 1 sentence of reasoning followed by YES or NO.\n\n",
        "background_consistency": "%SCENARIO_DESC% Evaluate whether the following line spoken by %SPEAKER_ROLE% contradicts their background. Provide 1 sentence explaining your reasoning based on the background, followed by YES if the line contradicts the given background of %SPEAKER_ROLE%, and NO if it is consistent with the provided background of %SPEAKER_ROLE%. %SPEAKER_ROLE%'s background is described as follows:\n%SPEAKER_BACKSTORY%\n %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n Provide 1 sentence of reason reasoning followed by YES or NO.\n\n",
        "combined_prompt_consistency": "%SCENARIO_DESC% Evaluate the intention behind the following line spoken by %SPEAKER_ROLE% and determine whether it contradicts their background. First, describe the interpreted intention of the statement, and whether or not it aligns with the given background of %SPEAKER_ROLE%. Then, answer YES if the line contradicts the given background of %SPEAKER_ROLE% or the intention does not align with the provided background, and answer NO if it does align with the provided background or the intention aligns with the background of %SPEAKER_ROLE%. Specifically, answer YES if %SPEAKER_ROLE%'s interaction style does not match, or if there are facts that don't match. %SPEAKER_ROLE%'s background is described as follows:\n%SPEAKER_BACKSTORY%\n %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n Provide your answer as 1 sentence explaining your reasoning based on the background and the interpreted intention, followed by YES or NO.\n\n",

        "pairwise_consistency":"%SCENARIO_DESC% For the following line spoken by %SPEAKER_ROLE%, answer YES if the line directly contradicts the provided line spoken by %LISTENER_ROLE%, and answer NO if the line does not contradict the provided line spoken by %LISTENER_ROLE%. %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n %LISTENER_ROLE% spoke the following line: \n%LISTENER_LINE%\n\n Answer YES if the line spoken by %SPEAKER_ROLE% contradicts the provided line spoken by %LISTENER_ROLE%, and answer NO if the line does not contradict the provided line spoken by %LISTENER_ROLE%, followed by 1 sentence of reasoning.\n\n",

        "backstory_test": "Based on the following background, generate a new fact-based multiple choice question with 5 choices addressed directly IN SECOND PERSON, along with its correct answer. Preface the question with 'Question:' and the answer with 'Answer:'.\n%SPEAKER_BACKSTORY%\n%PREVIOUS_QUESTIONS%",
        "answer_backstory": "You are %SPEAKER_ROLE%, and you are having a conversation with %LISTENER_ROLE%. Your background is:\n%SPEAKER_BACKSTORY%\n So far, the conversation is as below:\n%CONVERSATION%\n\n Based on your conversation above so far, answer the following multiple choice question.\n%BACKSTORY_QUESTION%\n",
        "grade_backstory": "As part of grading a test, determine whether the given answer %GIVEN_ANSWER% matches the following correct answer. Respond with either YES or NO.\nCorrect Answer: %CORRECT_ANSWER%\n"
    }

    def eval_prompt_consistency(conv_dict, both_agents=False):
        conv_dict['eval_prompt_consistency'] = []
        conv_dict['P1_prompt_consistency_score'] = 0
        conv_dict['P2_prompt_consistency_score'] = 0
        p1_utterances = 0
        p2_utterances = 0

        pturn = conv_dict["pturn"]
        for line in conv_dict["conversation"]:
            line_number = line[0]
            convo_line = line[1]
            if pturn == 1:
                prompt = eval_prompts["combined_prompt_consistency"].replace("%SCENARIO_DESC", prompts["agent1_prompt"]) \
                                                                    .replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                                                    .replace("%SPEAKER_BACKSTORY%", conv_dict["P1"]) \
                                                                    .replace("%SPEAKER_LINE%", convo_line)
                if config.get('verbose', False):
                    print(prompt)
                output = completion_create(config['eval_model'], config, prompt)
                conv_dict['eval_prompt_consistency'].append((line_number, output))
                if "YES" not in output:  # no contradiction
                    conv_dict['P1_prompt_consistency_score'] += 1
                p1_utterances += 1
                pturn = 2
            elif pturn == 2:
                if both_agents:
                    prompt = eval_prompts["combined_prompt_consistency"].replace("%SCENARIO_DESC", prompts["agent2_prompt"]) \
                                                                        .replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                                                                        .replace("%SPEAKER_BACKSTORY%", conv_dict["P2"]) \
                                                                        .replace("%SPEAKER_LINE%", convo_line)
                    if config.get('verbose', False):
                        print(prompt)
                    output = completion_create(config['eval_model'], config, prompt)
                    conv_dict['eval_prompt_consistency'].append((line_number, output))
                    if "YES" not in output:  # no contradiction
                        conv_dict['P2_prompt_consistency_score']+= 1
                    p2_utterances += 1
                pturn = 1

        if p1_utterances > 0:
            conv_dict['P1_prompt_consistency_score'] /= p1_utterances
        if p2_utterances > 0:
            conv_dict['P2_prompt_consistency_score'] /= p2_utterances

        if config.get('verbose', False):
            print(conv_dict)
        return conv_dict
    # Replacement for (2) and (4), evaluates whether each pair of lines in the conversation is consistent with each other

    import utils

    config.update(config_llm)
    prompts = config_therapy
    for conversation in tqdm(conversations[:10]):
        conversation = eval_prompt_consistency(conversation, both_agents=False)

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=4)

    write_stats(filepath)

    

if __name__ == "__main__":
    # Create the dialog
    # create_dialog()

    # Run evaluations
    # filepath = write_file
    filepath = "/mmfs1/home/donoclay/socialrl/donoclay/consistency_LLMs/therapy/exp/04.26.25/Mistral-7B-Instruct_0_04-26-21-49-54.json"
    run_evals(filepath)