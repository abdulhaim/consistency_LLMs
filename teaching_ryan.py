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


# In[6]:

# %env CUDA_VISIBLE_DEVICES=0


# In[2]:# In[4]:


# In[3]:


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


# In[4]:


personas = [
  # Elementary
  {
    "grade_level": "elementary school",
    "description": (
      "As an elementary school student with a Narrative learning style, I absorb new concepts best when they’re told as engaging mini-stories. "
      "In dialogue, I ask for short anecdotes that turn any abstract idea into a vivid tale with characters, a clear sequence, and an emotional hook. "
      "Stories help me remember causal links and keep details alive in my mind."
    )
  },
  {
    "grade_level": "elementary school",
    "description": (
      "As an elementary school student with a Kinesthetic learning style, I understand ideas by imagining myself performing them. "
      "In conversation, I ask you to guide me through a pretend play-through—verbally walking me step by step as if I’m enacting a simple experiment or physical process. "
      "This imagined movement helps me anchor concepts in ‘muscle memory’ even though we’re only talking."
    )
  },
  {
    "grade_level": "elementary school",
    "description": (
      "As an elementary school student with a Naturalistic learning style, I connect best when content is tied to the natural world through vivid imagery. "
      "In dialogue, I ask you to compare topics—like atomic structure—to things I observe outdoors, such as tree rings or bird migrations. "
      "These verbal nature metaphors make new information feel familiar and alive."
    )
  },
  {
    "grade_level": "elementary school",
    "description": (
      "As an elementary school student with an Experiential learning style, I learn by mentally simulating real-world tasks. "
      "In conversation, I ask you to walk me through building or testing something—describing each step as if I’m doing it. "
      "That imagined ‘doing’ makes concepts concrete, even though we remain in chat."
    )
  },
  {
    "grade_level": "elementary school",
    "description": (
      "As an elementary school student with a Creative-Divergent learning style, I thrive on brainstorming multiple possibilities. "
      "In dialogue, I propose ‘what if’ scenarios—like alternative endings or playful twists on a concept—and talk through each idea. "
      "Verbal brainstorming reveals fresh patterns and sparks my imagination."
    )
  },

  # Middle
  {
    "grade_level": "middle school",
    "description": (
      "As a middle school student with a Visual-Spatial learning style, I think in mental images and diagrams. "
      "In conversation, I ask you to ‘paint’ word-pictures—step-by-step descriptions of scenes or flows—so I can build a clear mental map. "
      "That verbal imagery helps me organize information spatially in my mind."
    )
  },
  {
    "grade_level": "middle school",
    "description": (
      "As a middle school student with an Auditory learning style, I internalize knowledge through sound and speech. "
      "In dialogue, I ask you to restate key points in different rhythms or tones, and I repeat them back to reinforce my memory. "
      "Hearing and echoing concepts in conversation makes them stick."
    )
  },
  {
    "grade_level": "middle school",
    "description": (
      "As a middle school student with a Logical-Mathematical learning style, I seek numerical patterns and rule-based reasoning. "
      "In dialogue, I pose ‘what-if’ questions—‘If X doubles, what changes?’—and we talk through each scenario using simple calculations. "
      "Quantitative hypotheticals build my systematic understanding."
    )
  },
  {
    "grade_level": "middle school",
    "description": (
      "As a middle school student with an Analytical-Argument learning style, I dissect arguments and causal chains. "
      "In conversation, I ask targeted ‘why’ and ‘how’ questions about each step, construct mini flow-charts aloud, and verify the logic with you. "
      "This structured debate hones my precision in reasoning."
    )
  },
  {
    "grade_level": "middle school",
    "description": (
      "As a middle school student with a Verbal-Linguistic learning style, I learn through rich language and writing. "
      "In dialogue, I request carefully worded definitions, paraphrase ideas in my own words, and craft mnemonic rhymes on the spot. "
      "Talking through ideas in text-like sentences and playing with words helps me remember precisely."
    )
  },
  {
    "grade_level": "middle school",
    "description": (
      "As a middle school student with a Technology-Enhanced learning style, I thrive on conversational simulations of digital tools. "
      "In dialogue, I ask you to describe how a virtual model might respond as we adjust parameters, or to role-play a flashcard quiz verbally. "
      "These imagined tech interactions keep me engaged without leaving our chat."
    )
  },
  {
    "grade_level": "middle school",
    "description": (
      "As a middle school student with a Mnemonic learning style, I anchor facts with memory aids. "
      "In dialogue, I ask for catchy acronyms, rhymes, or vivid mental images—then recite them back. "
      "That verbal encoding makes complex lists or steps easy to retrieve."
    )
  },
  {
    "grade_level": "middle school",
    "description": (
      "As a middle school student with an Emotional learning style, I connect through feelings and empathy. "
      "In conversation, I ask you to frame concepts in human-centered narratives that highlight emotional stakes. "
      "These emotionally rich verbal stories make ideas memorable and meaningful."
    )
  },

  # High School
  {
    "grade_level": "high school",
    "description": (
      "As a high school student with a Collaborative learning style, I excel in multi-voice discussions. "
      "In dialogue, I invite hypothetical peers into our chat—debating viewpoints, role-playing characters, or comparing interpretations. "
      "That social exchange refines my understanding."
    )
  },
  {
    "grade_level": "high school",
    "description": (
      "As a high school student with an Interpersonal learning style, I flourish in one-on-one exchanges. "
      "In conversation, I engage deeply with a single partner—asking questions, providing feedback, and co-constructing ideas through back-and-forth talk."
    )
  },
  {
    "grade_level": "high school",
    "description": (
      "As a high school student with a Reflective learning style, I pause and summarize before responding. "
      "In dialogue, I restate points in my own words, journal key ideas mentally, and then ask precise follow-ups. "
      "This verbal reflection clarifies gaps and deepens comprehension."
    )
  },
  {
    "grade_level": "high school",
    "description": (
      "As a high school student with a Metaphorical learning style, I anchor concepts in analogies. "
      "In dialogue, I ask you to compare subjects to familiar scenarios—‘It’s like X because…’—and we talk through how well the metaphor holds. "
      "Testing analogies verbally helps me translate abstract ideas into relatable terms."
    )
  },
  {
    "grade_level": "high school",
    "description": (
      "As a high school student with an Intrapersonal learning style, I connect content to my own values. "
      "In dialogue, I ask how topics relate to my goals or experiences and share personal reflections aloud. "
      "That self-referential talk makes learning relevant and motivating."
    )
  },
  {
    "grade_level": "high school",
    "description": (
      "As a high school student with a Problem-Based learning style, I tackle hypothetical real-world scenarios in talk. "
      "In dialogue, I propose case studies—like designing a sustainable system—and we walk through each decision together. "
      "Verbal scenario-based reasoning shows me practical applications of theory."
    )
  },
  {
    "grade_level": "high school",
    "description": (
      "As a high school student with a Trial-and-Error learning style, I learn by mentally testing ideas. "
      "In dialogue, I suggest imagined experiments—‘Let’s tweak this variable and see what happens’—and we discuss the outcomes. "
      "Using mistakes as discussion points builds discovery-based understanding."
    )
  },
  {
    "grade_level": "high school",
    "description": (
      "As a high school student with a Conceptual learning style, I focus on verbal mapping of frameworks. "
      "In dialogue, I request thematic overviews—described step by step—and we discuss how each piece fits into the big picture. "
      "Building mental models in talk deepens my flexible understanding."
    )
  },

  # College
  {
    "grade_level": "college",
    "description": (
      "As a college student with a Theoretical learning style, I probe abstract frameworks in conversation. "
      "In dialogue, I challenge you to trace ideas back to their assumptions, compare theoretical models, and debate implications. "
      "This verbal inquiry drives deep synthesis."
    )
  },
  {
    "grade_level": "college",
    "description": (
      "As a college student with a Research-Oriented learning style, I learn by interrogating studies in chat. "
      "In conversation, I ask for summaries of current research, discuss methods and controls, and role-play peer-review feedback. "
      "Critically evaluating evidence through talk builds an evidence-based grasp."
    )
  },
  {
    "grade_level": "college",
    "description": (
      "As a college student with an Integrative learning style, I weave ideas together verbally. "
      "In conversation, I ask for cross-topic syntheses—connecting historical, artistic, and scientific themes—and discuss their intersections step by step. "
      "This systems-level perspective helps me approach complex questions creatively."
    )
  },
  {
    "grade_level": "college",
    "description": (
      "As a college student with a Structured learning style, I excel on verbal outlines and modules. "
      "In dialogue, I ask for hierarchical breakdowns—numbered lists, staged explanations, and schematic overviews—before diving into details."
    )
  },
  {
    "grade_level": "college",
    "description": (
      "As a college student with a Solitary learning style, I prefer self-guided dialog prompts. "
      "In our conversation, I request personalized questions and silent think-time before sharing my conclusions, using chat as a safe space for independent reflection."
    )
  },
  {
    "grade_level": "college",
    "description": (
      "As a college student with an Adaptive learning style, I shift strategies based on what works. "
      "In dialogue, I monitor which verbal approaches—stories, logic puzzles, analogies—help me most and ask to switch accordingly. "
      "This dynamic, metacognitive talk ensures I absorb concepts through the most effective modality."
    )
  }
]


# In[5]:


import json
import shutil
import os

config_role = {
    "agent1_role": "Teacher",
    "agent2_role": "Student",
    "agent1_prompt": "You are a teacher whose goal is to guide a student through learning about %SUBJECT%. You have a preferred way to teach the student. The student is in %ROLE% so make sure to teach them at their level. ",
    "agent2_prompt": "You are a student in %ROLE% in conversation with a teacher who will teach you %SUBJECT%. You like to learn in the following way:\n%SPEAKER_BACKSTORY%.\nMake sure to not only ask questions but also demonstrate your knowledge.",
    'reminder_prompt': "Keep your response very brief — 2 sentences or less. Do NOT repeat anything you've already said.\n",
    'eval_prompts': {
    'prompt_consistency': """You are evaluating whether the following utterance by %SPEAKER_ROLE% is consistent with their given background. Answer YES if the line directly contradicts any part of the background—this includes contradictions in facts, behavior, beliefs, emotional stance, or conversational strategy. Answer NO if the line introduces new details, but those details are **plausibly aligned** with the background. Be strict and literal in your evaluation: if the line violates the persona’s stated traits (e.g., avoids vulnerability, keeps answers short, avoids discussing personal topics), mark it as a contradiction even if the violation is subtle. Background for %SPEAKER_ROLE%: %SPEAKER_BACKSTORY% Line spoken by %SPEAKER_ROLE%: %SPEAKER_LINE%  Answer YES if the line contradicts the background, and NO if it does not. Then, explain your answer in one sentence. Be precise and avoid vague justification."""}}

os.makedirs("education", exist_ok=True)
with open("education/config_education.json", "w", encoding="utf-8") as f:
    json.dump(config_role, f, indent=4)


# In[6]:


with open('./config/education/personas_education_master.json', 'r') as f:
    conversation_prompts = json.load(f)
conversation_prompts[0]['background_info'].keys()


# In[7]:


topic_list = []
for convo_prompt in conversation_prompts:
    topic_prompt = convo_prompt["background_info"]["topic"]
    topic_list.append(topic_prompt)


# In[8]:


llms = ["Llama-3.1-8B-Instruct", "gpt-4o-mini", "Qwen2.5-3B-Instruct", "Llama-3.1-8B", "Mistral-7B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct", "phi-3.5-mini-instruct"]
        
config_llm = {'agent1_model': 'mistral-instruct',
             'agent2_model': 'mistral-instruct',
             'eval_model': 'Llama-3.1-70B-Instruct',
             'iterations': 10,
             'verbose': False,
             'write': True,
            #  'fp8': True,
             'convo_length_limit': 10,
             'max_tokens': 256,
             'gpus': 1,
             'seed': 0,
             'task_name': 'Education',
             'model_dir': "/raid/users/ryan_cheng2/models"}

with open("education/Llama-3.1-8B-Instruct.json", "w", encoding="utf-8") as f:
    json.dump(config_llm, f, indent=4)


# In[9]:


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

def generate_response(agent_model, expected_role, other_role, config_llm, prompt, max_retries=10):
    count_retries = 0 
    role_confused = True
    while count_retries<max_retries:
        response = completion_create(agent_model, config_llm, prompt)
        print("Expected Role", expected_role)
        role_confused = is_role_confused(response, other_role)
        count_retries+=1
        if not is_role_confused(response, other_role):
            return clean_role_prefix(response, expected_role)
            
    return clean_role_prefix(response, expected_role)

def generate_conversation(config_llm, p1, p2, p1_name, p2_name, subject, role, pturn=1):
    stats['P1'] = p1
    stats['P2'] = p2
    stats["topic"] = subject
    stats["grade"] = role
    stats['pturn'] = pturn
    round_num = 0
    while round_num < config_llm['convo_length_limit']:
        conversation = ("".join([turn[1] if isinstance(turn, tuple) else turn for turn in stats["conversation"]]) if len(stats["conversation"]) != 0 else "You are starting the conversation.\n")

        if pturn == 1:
            prompt = config_role["agent1_prompt"]
            pturn = 2
            if config_llm["verbose"]:
                print(prompt)
                print()

            if round_num!=0: 
                prompt+= "Your conversation with the student so far is below:\nConversation:\n%CONVERSATION%"
                
            if round_num >=config_llm['convo_length_limit']*2-11 and round_num<=config_llm['convo_length_limit']*2-1:
                prompt+= "You have " + str((config_llm['convo_length_limit']-round_num)//2) + " rounds left." + "Make sure to conclude the conversation as your near the end."

            elif round_num>config_llm['convo_length_limit']*2-1:
                prompt+= "This is your concluding line in the conversation."

            if round_num!=0: 
                prompt+= "Continue the conversation with the student. Remember you are the teacher. "
                
            prompt += config_role["reminder_prompt"]
            prompt+="%SPEAKER_ROLE%:"
            prompt = prompt.replace("%SPEAKER_ROLE%", config_role["agent1_role"]) \
                   .replace("%LISTENER_ROLE%", config_role["agent2_role"]) \
                    .replace("%ROLE%", role) \
                   .replace("%SUBJECT%", subject) \
                   .replace("%CONVERSATION%", conversation)
            
            response = generate_response(config_llm['agent1_model'], config_role["agent1_role"], config_role["agent2_role"], config_llm, prompt)
            stats["conversation"].append((round_num, f"{config_role["agent1_role"]}: " + response + "\n"))
        
        else:
            prompt = config_role["agent2_prompt"]
            pturn = 1    
            if config_llm["verbose"]:
                print(prompt)
                print()

            if round_num!=0: 
                prompt+= "Your conversation with the teacher so far is below:\nConversation:\n%CONVERSATION%"
            if round_num >=config_llm['convo_length_limit']*2-11 and round_num<=config_llm['convo_length_limit']*2-1:
                prompt+= "You have " + str((config_llm['convo_length_limit']-round_num)//2) + " rounds left." + "Make sure to conclude the conversation as your near the end."
            elif round_num>config_llm['convo_length_limit']*2-1:
                prompt+= "This is your concluding line in the conversation."

            if round_num!=0: 
                prompt+= "Continue the conversation with the teacher. Remember you are the student. "

            prompt += config_role["reminder_prompt"]
            
            prompt+="%SPEAKER_ROLE%:"
            prompt = prompt.replace("%SPEAKER_ROLE%", config_role["agent2_role"]) \
               .replace("%LISTENER_ROLE%", config_role["agent1_role"]) \
               .replace("%SPEAKER_BACKSTORY%", p2) \
                .replace("%ROLE%", role) \
               .replace("%SUBJECT%", subject) \
               .replace("%CONVERSATION%", conversation)
            
            response = generate_response(config_llm['agent2_model'], config_role["agent2_role"], config_role["agent1_role"], config_llm, prompt)
            stats["conversation"].append((round_num, f"{config_role["agent2_role"]}: " + response + "\n"))
            
        round_num += 1

    stats["rounds"] = round_num
    if config_llm['verbose']:
        print(stats["conversation"])
    return stats.copy()

def reset_stats():
    stats_template = {
        "task_name": config_llm['task_name'],
        "topic": "",
        "grade": "",
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


# In[10]:


import os
import random
from datetime import datetime
import utils
utils.config = config_llm

current_date = str(datetime.now().strftime("%m.%d.%y"))
output_dir = f"education/exp/{current_date}"
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


# In[11]:


import random

# 1. Build a “persona pool” of length 100
pool = personas * (100 // len(personas)) \
     + random.sample(personas, 100 % len(personas))
random.shuffle(pool)

# 2. Sample 100 topics *with* replacement
topic_choices = random.choices(topic_list, k=100)

# 3. Zip them into 100 pairs
persona_final = list(zip(topic_choices, pool))
assert len(persona_final) == 100


# In[12]:


with open("education/config_education_personas.json", "w", encoding="utf-8") as f:
    json.dump(persona_final, f, indent=4)

if __name__ == "__main__":
  index_offset = load_stats_file(write_file)
  conversations = []    
  lengths = [10, 20, 40, 60]
  # lengths = [40]
  count = 0 
  for i in range(1):
      for topic, persona_item in tqdm(persona_final[index_offset+1:]):
          count+=1
          print(count)
          background = persona_item["description"]
          grade = persona_item["grade_level"]
          for convo_length in lengths:
              config_llm['convo_length_limit'] = convo_length
              reset_stats()
              conversation = generate_conversation(
                  config_llm,
                  "", 
                  background, 
                  "Teacher", 
                  "Student", 
                  topic, 
                  grade, 
                  pturn=1
              )

              # conversation_eval = consistency_eval.eval_prompt_consistency(conversation, agents=(2,))
              # conversation_eval = consistency_eval.eval_index_consistency(conversation_eval, agents=(2,))
              print(conversation)
              # print(conversation_eval)
              # conversations.append(conversation_eval)
              stats['index'] = index_offset
              stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
              write_stats(write_file)
              index_offset += 1
