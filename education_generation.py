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
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

def generate_conversation(background_info_student, topic, pturn=1):
    '''
    background_info: a dictionary containing the keys 'topic', 'student_preferences', 'teacher_preferences', 'student_reactions', 'teacher_reactions'
    '''
    teacher_preferences = "You are continuing the conversation as the teacher. "
    student_preferences = prompts['student_preferences'].replace("%STUDENT_PREFERENCE%", background_info_student['student_preference']) \
                                                        .replace("%STUDENT_REACTIONS%", background_info_student['student_reaction'])
    
    dialogue_prompt = prompts["dialogue_prompt"].replace("%STUDENT_BACKGROUND%", background_info_student['student_background']) \
                                                .replace("%TOPIC%", topic) \

    stats['P1'] = teacher_preferences
    stats['P2'] = student_preferences
    stats['topic'] = topic
    stats['pturn'] = pturn

    round_num = 0
    while round_num < config['convo_length_limit'] and '<END>' not in "".join(stats["conversation"]):
        conversation = ("".join(stats["conversation"]) if len(stats["conversation"]) != 0 else "[You are starting the conversation.]\n")
        if pturn == 1: # teacher
            prompt = dialogue_prompt.replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                    .replace("%PREFERENCES%", teacher_preferences) \
                                    .replace("%CONVERSATION%", conversation)
            pturn = 2
            if config["verbose"]:
                print(prompt)
                print()
            model_response = completion_create(config['agent1_model'], config, prompt)
            model_response = split_conversation(f"{prompts['agent1_role']}: " + model_response, prompts["agent1_role"], prompts["agent2_role"])[0] # filters out dialogues with more than one response
            stats["conversation"].append(model_response + "\n")
        else: # student
            prompt = dialogue_prompt.replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                                    .replace("%PREFERENCES%", student_preferences) \
                                    .replace("%CONVERSATION%", conversation)
            pturn = 1    
            if config["verbose"]:
                print(prompt)
                print()
            model_response = completion_create(config['agent2_model'], config, prompt)
            model_response = split_conversation(f"{prompts['agent2_role']}: " + model_response, prompts["agent1_role"], prompts["agent2_role"])[0] # filters out dialogues with more than one response
            stats["conversation"].append(model_response + "\n")
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
        "topic": "",
        "student_background": "",
        "student_preference": "",
        "student_reaction": "",
        "conversation": [],
        "pturn": 0,
        "index": -1,
        "timestamp": "",
        "rounds": 0,
        'conversation_only': True,
        'model1': '',
        'model2': ''
    }
    for key, value in stats_template.items():
        stats[key] = value

def generate_conversation_old(background_info_student, background_info_teacher, topic, pturn=1):
    '''
    Generate with teacher preferences, deprecated.
    background_info: a dictionary containing the keys 'topic', 'student_preferences', 'teacher_preferences', 'student_reactions', 'teacher_reactions'
    '''
    teacher_preferences = prompts['teacher_preferences'].replace("%TEACHER_PREFERENCE%", background_info_teacher['teacher_prefrences']).replace("%TEACHER_REACTIONS%", background_info_teacher['teacher_reactions'])
    student_preferences = prompts['student_preferences'].replace("%STUDENT_PREFERENCE%", background_info_student['student_prefrences']).replace("%STUDENT_REACTIONS%", background_info_student['student_reactions'])
    
    stats['P1'] = teacher_preferences
    stats['P2'] = student_preferences
    stats['topic'] = topic
    stats['pturn'] = pturn

    round_num = 0
    while round_num < config['convo_length_limit'] and '<END>' not in "".join(stats["conversation"]):
        conversation = ("".join(stats["conversation"]) if len(stats["conversation"]) != 0 else "[You are starting the conversation.]\n")
        if pturn == 1: # teacher
            prompt = prompts["dialogue_prompt"].replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                               .replace("%TOPIC%", topic) \
                                               .replace("%PREFERENCES%", teacher_preferences) \
                                               .replace("%CONVERSATION%", conversation)
            pturn = 2
            if config["verbose"]:
                print(prompt)
                print()
            model_response = completion_create(config['agent1_model'], config, prompt)
            model_response = split_conversation(f"{prompts['agent1_role']}: " + model_response, prompts["agent1_role"], prompts["agent2_role"])[0] # filters out dialogues with more than one response
            stats["conversation"].append(model_response + "\n")
        else: # student
            prompt = prompts["dialogue_prompt"].replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                                               .replace("%TOPIC%", topic) \
                                               .replace("%PREFERENCES%", student_preferences) \
                                               .replace("%CONVERSATION%", conversation)
            pturn = 1    
            if config["verbose"]:
                print(prompt)
                print()
            model_response = completion_create(config['agent2_model'], config, prompt)
            model_response = split_conversation(f"{prompts['agent2_role']}: " + model_response, prompts["agent1_role"], prompts["agent2_role"])[0] # filters out dialogues with more than one response
            stats["conversation"].append(model_response + "\n")
        round_num += 1

    stats["rounds"] = round_num
    if config['verbose']:
        print(stats["conversation"])
    return stats.copy()

def main(argv):
    global prompts
    init()

    # Load prompts
    with open('config/education/prompts.json', 'r') as f:
        prompts = json.load(f)

    # Load conversation data
    with open('data/education/conversations_train1.json', 'r') as f:
        conversation_prompts = json.load(f)
    
    write_file = (
        f"data/education/exp/"
        f"{config['agent1_model']}_{config['seed']}.json"
    )
    config["task_name"] = "Education"

    index_offset = load_stats_file(write_file)
    if index_offset > config['iterations']:
    #     setup_vllm()
    # else:
        print('Info: Skipping file!')
    
    # max of 1000 randomly selected conversations, extend size if you want more
    np.random.seed(0)
    random_topic_i = np.random.choice(len(conversation_prompts), size=1000, replace=False).astype(int)
    np.random.seed(0)
    random_indices = np.random.choice([0,1], size=3000, replace=True).astype(int).reshape(-1, 3)

    student_background = ["elementary school student", "high school student", "graduate student"]
    student_preferences = ["interactive", "story", "lecture"]
    student_reactions = ["rude", "anxious", "distant"]
    for iteration in tqdm(range(config['iterations']-index_offset)):
        if index_offset < config['iterations']:
            reset_stats()

            background_i, pref_i, reaction_i = random_indices[index_offset]
            topic_i = random_topic_i[index_offset]

            pturn = index_offset % 2 + 1
            topic = conversation_prompts[topic_i]['background_info']['topic']
            stats['student_background'] = student_background[1]
            stats['student_preference'] = student_preferences[pref_i]
            stats['student_reaction'] = student_reactions[reaction_i]

            student_dict = {
                "student_background": student_background[1],
                "student_preference": prompts[student_preferences[pref_i]],
                "student_reaction": prompts[student_reactions[reaction_i]],
            }
            generate_conversation(student_dict, topic)
            
            stats['model1'] = config['agent1_model']
            stats['model2'] = config['agent2_model']
            stats['index'] = (index_offset if config['write'] else -1)
            stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            index_offset += 1
            if config['write']:
                write_stats(write_file)

if __name__ == '__main__':
    app.run(main)