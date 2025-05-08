#!/usr/bin/env python

<<<<<<< HEAD
=======

>>>>>>> main
# Imports 
import os
import logging
import subprocess
import torch
import glob
import re
import json
<<<<<<< HEAD
import shutil
import time
import pickle
import random

=======
import random
import time
import pickle
>>>>>>> main
from absl import app, flags
from tqdm import tqdm
from datetime import datetime
import openai
from openai import OpenAI
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
<<<<<<< HEAD
=======

from utils import *
import utils
>>>>>>> main
try:
    from vllm import LLM, SamplingParams
    import ray
except ImportError:
    pass
    
<<<<<<< HEAD
# File Imports 
from utils import *
import utils
    
=======
>>>>>>> main
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

<<<<<<< HEAD
=======

>>>>>>> main
## Utils 
def get_freest_cuda_device():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, encoding='utf-8')
    memory_free = [int(x) for x in result.stdout.strip().split('\n')]
    return memory_free.index(max(memory_free))


<<<<<<< HEAD
def setup_():
=======
def setup():
>>>>>>> main
    best_gpu = get_freest_cuda_device()
    device = torch.device(f"cuda:{best_gpu}")
    print(f"Using GPU: {device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

    with open(os.path.abspath('../openai_key'), 'r') as f:
        utils.client = OpenAI(api_key=f.read().rstrip('\n'))

<<<<<<< HEAD
    with open("therapy/Llama-3.1-8B-Instruct.json", "w", encoding="utf-8") as f:
        json.dump(config_llm, f, indent=4)
    
    with open("therapy/config_therapy_personas.json", "r", encoding="utf-8") as f:
        personas_therapy = json.load(f)


=======
>>>>>>> main
def create_therapy_personas()
    with open("../token.txt", "r") as f:
        token = f.read().strip()

    from huggingface_hub import login
    login(token=token)

    
    from datasets import load_dataset
    ds = load_dataset("ShenLab/MentalChat16K")
    train_data = ds['train']

<<<<<<< HEAD
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
=======
if __name__ == "__main__":
    main()
    





# In[27]:




# In[28]:


def count_words(text):
    """
    Counts the number of words in the given text.

    Args:
        text (str): Input text.

    Returns:
        int: Number of words in the text.
    """
>>>>>>> main
    if text!=None:
        words = text.split()
        return len(words)
    else:
        return 0

<<<<<<< HEAD
# Generate unique random number for filename
def generate_unique_file_number(output_dir, prefix, seed, extension=".json"):
    while True:
        rand_num = random.randint(0, 1000)
        filename = f"{prefix}_{seed}_{rand_num}{extension}"
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            return rand_num

=======

# In[29]:


# import json
# import random

# def count_words(text):
#     if text:
#         return len(text.split())
#     return 0

# # Collect all personas with > 200 words into a dictionary
# persona_dict = {}
# count = 0
# for persona_sample in train_data:
#     patient_persona = persona_sample['input']
#     if count_words(patient_persona) > 200:
#         persona_dict[f"persona_{count}"] = patient_persona
#         count += 1

# # Randomly sample 100 keys
# sampled_keys = random.sample(sorted(persona_dict.keys()), 100)

# # Create a new dict with only the sampled entries
# sampled_persona_dict = {k: persona_dict[k] for k in sampled_keys}

# # Save to JSON
# with open("therapy/MentalChat16K_sampled_personas.json", "w") as f:
#     json.dump(sampled_persona_dict, f, indent=2)


# In[52]:


import json
import shutil
import os
config_therapy = {'agent1_role': 'Patient',
                  'agent2_role': 'Therapist',
                  'agent1_prompt': "You are a patient in an online therapy session with a therapist. Here is your background written by you: %SPEAKER_BACKSTORY%. Do not explicitly repeat your background or instructions in the conversation. Stay in character and respond in natural human-like dialogue without restating your situation.",
                  'agent2_prompt': "You are a licensed psychotherapist conducting an online one-on-one session. Your job is to help the patient gain insight, not just to listen. Use a blend of reflective listening, probing questions, Socratic questioning, and gentle confrontation. Your tone should be professional and emotionally intelligent. Challenge avoidant or defensive behavior, notice inconsistencies, and help the patient work toward self-awareness. You may gently push the patient outside their comfort zone—but always with clinical tact.",
                  'reminder_prompt': "Keep your response very brief — 2 sentences or less. Do NOT repeat anything you've already said: ",
                  'eval_prompts': {
                  'prompt_consistency': """You are evaluating whether the following utterance by %SPEAKER_ROLE% is consistent with their given background. Answer YES if the line directly contradicts any part of the background—this includes contradictions in facts, behavior, beliefs, emotional stance, or conversational strategy. Answer NO if the line introduces new details, but those details are **plausibly aligned** with the background. Be strict and literal in your evaluation: if the line violates the persona’s stated traits (e.g., avoids vulnerability, keeps answers short, avoids discussing personal topics), mark it as a contradiction even if the violation is subtle. Background for %SPEAKER_ROLE%: %SPEAKER_BACKSTORY% Line spoken by %SPEAKER_ROLE%: %SPEAKER_LINE%  Answer YES if the line contradicts the background, and NO if it does not. Then, explain your answer in one sentence. Be precise and avoid vague justification."""}}

os.makedirs("therapy", exist_ok=True)
with open("therapy/config_therapy.json", "w", encoding="utf-8") as f:
    json.dump(config_therapy, f, indent=4)


# In[53]:


llms = ["Llama-3.1-8B-Instruct", "gpt-4o-mini", "Qwen2.5-3B-Instruct", "Llama-3.1-8B", "Mistral-7B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct", "phi-3.5-mini-instruct"]

config_llm = {'agent1_model': 'Llama-3.1-8B-Instruct',
             'agent2_model': 'Llama-3.1-8B-Instruct',
             'eval_model': 'Llama-3.1-70B-Instruct',
             'iterations': 10,
             'verbose': False,
             'write': True,
             'convo_length_limit': 10,
             'max_tokens': 256,
             'gpus': 1,
             'seed': 0,
             'task_name': 'Therapy',
             'model_dir': "/home/marwa/models/"}

with open("therapy/Llama-3.1-8B-Instruct.json", "w", encoding="utf-8") as f:
    json.dump(config_llm, f, indent=4)


# In[9]:


# personas_therapy = [
#   {
#     "name": "Melissa",
#     "age": 35,
#     "occupation": "Corporate Lawyer",
#     "condition": "Burnout, shame around seeking help",
#     "description": "I am a 35-year-old corporate lawyer going through chronic burnout. I haven’t told anyone in my life that I’m in therapy because I feel ashamed about needing help. It’s hard to admit I’m struggling.",
#     "strategy": "I try to avoid opening up too much. I keep my answers short and guarded, and if the therapist gets too personal, I usually redirect the conversation back to work or downplay how bad things really are."
#   },
#   {
#     "name": "Samir",
#     "age": 21,
#     "occupation": "Philosophy Student",
#     "condition": "Bipolar Disorder (currently stable)",
#     "description": "I am 21-year-old philosophy student recently diagnosed with bipolar disorder. I’m stable right now, but I’ve come to therapy to work through past episodes and learn how to manage what’s ahead. I tend to think about emotions like philosophical problems.",
#     "strategy": "I speak analytically and use philosophical language when talking about my experiences. I prefer discussing ideas over feelings, and I often avoid emotional language even when asked directly about how I feel."
#   },
#   {
#     "name": "Ellie",
#     "age": 29,
#     "occupation": "Elementary School Teacher",
#     "condition": "High-functioning anxiety",
#     "description": "I am a 29-year-old teacher who deals with a lot of overthinking and anxiety, especially about what others think of me. I tend to ramble when I’m nervous and I overshare without meaning to. I really want to get things 'right' in therapy.",
#     "strategy": "I talk a lot and jump between topics. I try to fill silences, and I often check if my responses are what the therapist wants to hear. I’m eager to please and sometimes share too much too fast."
#   },
#   {
#     "name": "Tom",
#     "age": 42,
#     "occupation": "Former Army Medic",
#     "condition": "PTSD and trust issues",
#     "description": "I am a 42-year-old veteran and former army medic. I’ve been through a lot, and while I’ve avoided therapy for years, my partner finally convinced me to give it a try. I don’t really trust the process yet.",
#     "strategy": "I keep my guard up. I’m skeptical about therapy and tend to shut down emotional questions. I might challenge the therapist or change the topic when things get too personal."
#   },
#   {
#     "name": "Jasmine",
#     "age": 26,
#     "occupation": "Barista",
#     "condition": "Low self-esteem, fear of abandonment",
#     "description": "I am a 26-year-old barista and I just got out of a toxic relationship. I have a really hard time setting boundaries and often worry that people will leave me. I'm trying to work on that.",
#     "strategy": "I try hard to make the therapist like me. I mirror their language and avoid conflict. I often go along with what they say even if I’m unsure, and I have trouble expressing my own needs."
#   },
#   {
#     "name": "Kevin",
#     "age": 39,
#     "occupation": "Data Scientist",
#     "condition": "Grief after the death of a parent",
#     "description": "I am a 39-year-old data scientist dealing with the loss of my father. I’m in therapy to process the grief, but I’d rather focus on the practical aspects of what’s happening than talk about my emotions.",
#     "strategy": "I tend to stay detached and analytical. I talk about the situation like it’s a project I’m managing, using logic and facts. I avoid emotional reflection whenever possible."
#   },
#       {
#     "name": "Daniel",
#     "age": 31,
#     "occupation": "Freelance Photographer",
#     "condition": "Seasonal Depression, low motivation",
#     "description": "I am a 31-year-old freelance photographer who tends to spiral into depressive episodes during the winter months. Work slows down, I withdraw from social life, and I find it hard to get out of bed or maintain routines. Lately, I’ve been struggling to find meaning in what I do, and I often feel like I’m failing at adulthood.",
#     "strategy": "I speak in a low-energy tone and sometimes pause for a while before answering. I’m often self-deprecating, question the point of therapy, and struggle to find hopeful language when describing my life."
#   },
#   {
#     "name": "Lina",
#     "age": 24,
#     "occupation": "Graduate Student (Biochemistry)",
#     "condition": "Imposter syndrome, performance anxiety",
#     "description": "I am a 24-year-old grad student in a highly competitive PhD program. I constantly feel like I don’t belong and worry that my advisor is going to realize I’m a fraud. Even though I get praise sometimes, I never believe it’s sincere. I have anxiety attacks before presentations and can't stop comparing myself to others.",
#     "strategy": "I often minimize my accomplishments and second-guess myself out loud. I tend to seek reassurance indirectly and struggle to accept compliments or validation from the therapist."
#   },
#   {
#     "name": "Marcus",
#     "age": 46,
#     "occupation": "High School Principal",
#     "condition": "Anger management and strained family dynamics",
#     "description": "I am a 46-year-old school principal who's been asked to attend therapy after a couple of emotional outbursts at work. My spouse says I have trouble expressing feelings unless it’s anger. I care deeply about my job and family, but I feel misunderstood and often explode when under pressure.",
#     "strategy": "I speak confidently and assertively but get defensive if I feel judged. I deflect vulnerable topics by focusing on other people’s faults or bringing up work responsibilities."
#   },
#   {
#     "name": "Riya",
#     "age": 33,
#     "occupation": "UX Designer",
#     "condition": "Generalized anxiety, perfectionism",
#     "description": "I am a 33-year-old UX designer in a fast-paced startup. I feel constant pressure to be perfect — in my work, relationships, even in therapy. I make endless to-do lists but feel like I'm never doing enough. I lie awake at night thinking about what I forgot to do.",
#     "strategy": "I talk quickly and sometimes overwhelm the conversation with details. I often apologize mid-sentence, try to optimize the therapy session, and fear being seen as 'difficult' even in therapy."
#   },
#   {
#     "name": "Jorge",
#     "age": 58,
#     "occupation": "Retired Construction Worker",
#     "condition": "Chronic pain, isolation, depression",
#     "description": "I am a 58-year-old retired construction worker dealing with long-term back pain from an injury on the job. Since retiring, I feel like I’ve lost my sense of purpose. My kids have moved away, and some days I don’t talk to anyone at all. I miss feeling useful.",
#     "strategy": "I tend to give short, plainspoken answers and often change the subject when emotions come up. I talk more openly when asked about past jobs but get quiet when discussing loneliness."
#   },
#   {
#     "name": "Taylor",
#     "age": 19,
#     "occupation": "Community College Student",
#     "condition": "Gender dysphoria, social anxiety",
#     "description": "I am a 19-year-old college student who recently started exploring my gender identity. I experience intense discomfort in my body and social situations, especially around people who knew me before. I often feel invisible or hyper-visible — like I can’t do anything right.",
#     "strategy": "I’m cautious and slow to open up. I often hedge what I say with 'maybe' or 'I don’t know.' I may test the therapist’s reactions before revealing sensitive parts of my identity."
#   }
# ]


# In[10]:


# persona_prompt = """You are a helpful assistant that, given a patient persona description, crafts a coping strategy describing how that persona would talk to their therapist.

# Input: <Brief text describing the patient's core issue and behavior patterns>
# Output: <One to two sentences in first person, showing how this persona speaks or defends themselves in therapy>

# Example:
# Input: Struggles to build and maintain healthy relationships, feels anxious and rejected whenever conflicts arise, and doubts self-worth when friends distance themselves.
# Output: I speak guardedly about my feelings, hesitate before opening up, and redirect the conversation when conflict feels too personal.

# Example:
# Input: Overwhelmed by decision-making, fears making the 'wrong' choice and second-guesses every option.
# Output: I inundate the conversation with hypothetical scenarios and ask repeated clarifying questions to delay committing to any decision.

# Now process this new persona:
# Input: """

# personas_therapy = []
# for therapist_persona in sampled_persona_dict:
#     input_prompt = persona_prompt + sampled_persona_dict[therapist_persona] + "\nOutput: "
#     output = completion_create("gpt-4o-mini", config_llm, input_prompt)
#     print(output)
#     personas_therapy.append({"description": sampled_persona_dict[therapist_persona], "strategy": output})


# In[32]:


# with open("therapy/config_therapy_personas.json", "w", encoding="utf-8") as f:
#     json.dump(personas_therapy, f, indent=4)


# In[12]:


with open("therapy/config_therapy_personas.json", "r", encoding="utf-8") as f:
    personas_therapy = json.load(f)


# In[32]:


import re
>>>>>>> main

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
<<<<<<< HEAD
    

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
=======

def generate_therapy(config_llm, p1, p2, p1_name, p2_name, pturn=1):
>>>>>>> main
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

<<<<<<< HEAD
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
    






=======
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


# In[33]:


import os
import random
from datetime import datetime
import utils
utils.config = config_llm

current_date = str(datetime.now().strftime("%m.%d.%y"))
output_dir = f"therapy/exp/{current_date}"
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
write_file = os.path.join(output_dir, f"{config_llm['agent1_model']}_{config_llm['seed']}_{unique_num}.json")


# In[ ]:


index_offset = load_stats_file(write_file)
conversations = []    
lengths = [10, 20, 40, 60]
for i in range(1):
    for patient_dict in personas_therapy:
        for convo_length in lengths:
            config_llm['convo_length_limit'] = convo_length
            reset_stats()
            conversation = generate_therapy(
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


# In[34]:


with open("therapy/exp/04.24.25/Llama-3.1-8B-Instruct_0_250.json", "r", encoding="utf-8") as f:
    conversations = json.load(f)


# In[43]:
>>>>>>> main


eval_prompts = {
    "strategy_consistency": "%SCENARIO_DESC% Evaluate whether the following line spoken by %SPEAKER_ROLE% contradicts their background. First, describe the interpreted intention of the statement, and whether or not it aligns with the given background of %SPEAKER_ROLE%. Then, provide 1 sentence explaining your reasoning based on the strategy, followed by YES if the line contradicts the given background of %SPEAKER_ROLE%, and NO if it is consistent with the provided background of %SPEAKER_ROLE%. %SPEAKER_ROLE%'s strategy is described as follows:\n%SPEAKER_STRATEGY%\n %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n Provide 1 sentence of reasoning followed by YES or NO.\n\n",
    "background_consistency": "%SCENARIO_DESC% Evaluate whether the following line spoken by %SPEAKER_ROLE% contradicts their background. Provide 1 sentence explaining your reasoning based on the background, followed by YES if the line contradicts the given background of %SPEAKER_ROLE%, and NO if it is consistent with the provided background of %SPEAKER_ROLE%. %SPEAKER_ROLE%'s background is described as follows:\n%SPEAKER_BACKSTORY%\n %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n Provide 1 sentence of reason reasoning followed by YES or NO.\n\n",
    "combined_prompt_consistency": "%SCENARIO_DESC% Evaluate the intention behind the following line spoken by %SPEAKER_ROLE% and determine whether it contradicts their background. First, describe the interpreted intention of the statement, and whether or not it aligns with the given background of %SPEAKER_ROLE%. Then, answer YES if the line contradicts the given background of %SPEAKER_ROLE% or the intention does not align with the provided background, and answer NO if it does align with the provided background or the intention aligns with the background of %SPEAKER_ROLE%. %SPEAKER_ROLE%'s background is described as follows:\n%SPEAKER_BACKSTORY%\n %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n Provide your answer as 1 sentence explaining your reasoning based on the background and the interpreted intention, followed by YES or NO.\n\n",

    "pairwise_consistency":"%SCENARIO_DESC% For the following line spoken by %SPEAKER_ROLE%, answer YES if the line directly contradicts the provided line spoken by %LISTENER_ROLE%, and answer NO if the line does not contradict the provided line spoken by %LISTENER_ROLE%. %SPEAKER_ROLE% spoke the following line: \n%SPEAKER_LINE%\n\n %LISTENER_ROLE% spoke the following line: \n%LISTENER_LINE%\n\n Answer YES if the line spoken by %SPEAKER_ROLE% contradicts the provided line spoken by %LISTENER_ROLE%, and answer NO if the line does not contradict the provided line spoken by %LISTENER_ROLE%, followed by 1 sentence of reasoning.\n\n",

    "backstory_test": "Based on the following background, generate a new fact-based multiple choice question with 5 choices addressed directly IN SECOND PERSON, along with its correct answer. Preface the question with 'Question:' and the answer with 'Answer:'.\n%SPEAKER_BACKSTORY%\n%PREVIOUS_QUESTIONS%",
    "answer_backstory": "You are %SPEAKER_ROLE%, and you are having a conversation with %LISTENER_ROLE%. Your background is:\n%SPEAKER_BACKSTORY%\n So far, the conversation is as below:\n%CONVERSATION%\n\n Based on your conversation above so far, answer the following multiple choice question.\n%BACKSTORY_QUESTION%\n",
    "grade_backstory": "As part of grading a test, determine whether the given answer %GIVEN_ANSWER% matches the following correct answer. Respond with either YES or NO.\nCorrect Answer: %CORRECT_ANSWER%\n"
}


<<<<<<< HEAD
=======
# In[44]:


# def eval_prompt_consistency(conv_dict):
#     #assert 'eval_prompt_consistency' not in conv_dict # warn if we are replacing metrics we don't mean to overwrite
#     conv_dict['eval_prompt_consistency'] = []
#     conv_dict['P1_prompt_consistency_score'] = 0
#     p1_utterances = 0
#     pturn = conv_dict["pturn"]
#     for line in conv_dict["conversation"]:
#         line_number = line[0]
#         convo_line = line[1]
#         if pturn == 1:
#             prompt = config_therapy["eval_prompts"]["prompt_consistency"].replace("%SPEAKER_ROLE%", config_therapy["agent1_role"]) \
#                                                                           .replace("%SPEAKER_BACKSTORY%", conv_dict["P1"]) \
#                                                                           .replace("%SPEAKER_LINE%", convo_line)
#             if config_llm['verbose']:
#                 print(prompt)
#             output = completion_create(config_llm['eval_model'], config, prompt)
#             conv_dict['eval_prompt_consistency'].append((line_number, output))
#             if "YES" not in output: # no contradiction
#                 conv_dict['P1_prompt_consistency_score'] += 1
#             p1_utterances += 1
#             pturn = 2
#         elif pturn == 2:
#             pturn = 1
#     if p1_utterances > 0:
#         conv_dict['P1_prompt_consistency_score'] /= p1_utterances
#     print(conv_dict)

#     return conv_dict


# In[54]:


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


# In[ ]:


import utils

config = config_llm
prompts = config_therapy
for conversation in tqdm(conversations):
    conversation = eval_prompt_consistency(conversation, both_agents=False)

write_stats(write_file)


# In[40]:

>>>>>>> main

with open("therapy/exp/04.24.25/Llama-3.1-8B-Instruct_0_250_consistency1.json", "w", encoding="utf-8") as f:
    json.dump(conversations, f, indent=4)


<<<<<<< HEAD
=======
# In[ ]:


conversations


# In[ ]:





# In[ ]:





# In[ ]:


>>>>>>> main


