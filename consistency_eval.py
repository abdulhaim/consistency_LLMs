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


flags.DEFINE_string('task', 'Anthology', 'run metrics on a particular task, searching for files within the task folder (Anthology/default)')

# (1) Takes in dialog, takes in base prompt, checks inconsistencies with base prompt for each line and output

def eval_prompt_consistency(conv_dict):
    #assert 'eval_prompt_consistency' not in conv_dict # warn if we are replacing metrics we don't mean to overwrite
    conv_dict['eval_prompt_consistency'] = []
    conv_dict['P1_prompt_consistency_score'] = 0
    conv_dict['P2_prompt_consistency_score'] = 0
    p1_utterances = 0
    p2_utterances = 0
    pturn = conv_dict["pturn"]
    for line in conv_dict["conversation"]:
        if pturn == 1:
            prompt = prompts["eval_prompts"]["prompt_consistency"].replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                                                  .replace("%SPEAKER_BACKSTORY%", conv_dict["P1"]) \
                                                                  .replace("%SPEAKER_LINE%", line)
            if config['verbose']:
                print(prompt)
            output = completion_create(config['eval_model'], config, prompt)
            conv_dict['eval_prompt_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P1_prompt_consistency_score'] += 1
            p1_utterances += 1
            pturn = 2
        else:
            prompt = prompts["eval_prompts"]["prompt_consistency"].replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                                                                  .replace("%SPEAKER_BACKSTORY%", conv_dict["P2"]) \
                                                                  .replace("%SPEAKER_LINE%", line)
            if config['verbose']:
                print(prompt)
            output = completion_create(config['eval_model'], config, prompt)
            conv_dict['eval_prompt_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P2_prompt_consistency_score'] += 1
            p2_utterances += 1
            pturn = 1
    
    if p1_utterances > 0:
        conv_dict['P1_prompt_consistency_score'] /= p1_utterances
    if p2_utterances > 0:
        conv_dict['P2_prompt_consistency_score'] /= p2_utterances

# Replacement for (2) and (4), evaluates whether each pair of lines in the conversation is consistent with each other
def eval_pairwise_consistency(conv_dict):
    conv_dict['eval_pairwise_consistency'] = []
    conv_dict['P1_pairwise_consistency_score'] = 0
    conv_dict['P2_pairwise_consistency_score'] = 0
    p1_utterances = 0
    p2_utterances = 0
    conversation = conv_dict["conversation"]
    pturn1 = conv_dict["pturn"]
    pturn2 = 1 if pturn1 == 2 else 2
    debug = []
    for i, line1 in enumerate(conversation):
        for j, line2 in enumerate(conversation):
            if i >= j:  # Skip comparing a line with itself or previous lines
                continue
            listener_role = prompts["agent1_role"] if pturn2 == 1 else prompts["agent2_role"]
            if pturn1 == 1:
                prompt = prompts["eval_prompts"]["pairwise_consistency"].replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                                                        .replace("%LISTENER_ROLE%", listener_role) \
                                                                        .replace("%SPEAKER_LINE%", line1) \
                                                                        .replace("%LISTENER_LINE%", line2)
                debug.append(prompt)
                if config['verbose']:
                    print(prompt)
                output = completion_create(config['eval_model'], config, prompt)
                conv_dict['eval_pairwise_consistency'].append(output)
                if "YES" not in output:  # no contradiction
                    conv_dict['P1_pairwise_consistency_score'] += 1
                p1_utterances += 1
                pturn2 = 1 if pturn2 == 2 else 2
            else:
                prompt = prompts["eval_prompts"]["pairwise_consistency"].replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                                                                        .replace("%LISTENER_ROLE%", listener_role) \
                                                                        .replace("%SPEAKER_LINE%", line1) \
                                                                        .replace("%LISTENER_LINE%", line2)
                debug.append(prompt)
                if config['verbose']:
                    print(prompt)
                output = completion_create(config['eval_model'], config, prompt)
                conv_dict['eval_pairwise_consistency'].append(output)
                if "YES" not in output:  # no contradiction
                    conv_dict['P2_pairwise_consistency_score'] += 1
                p2_utterances += 1
                pturn2 = 1 if pturn2 == 2 else 2
        
        # Swap turns for i for the next iteration
        pturn1 = 1 if pturn1 == 2 else 2
        pturn2 = 1 if pturn1 == 2 else 2 # set pturn2 to the opposite of the new pturn1

    if p1_utterances > 0:
        conv_dict['P1_pairwise_consistency_score'] /= p1_utterances
    if p2_utterances > 0:
        conv_dict['P2_pairwise_consistency_score'] /= p2_utterances
    return debug

# (2) Takes in dialog, checks inconsistencies with every line henceforth 
def eval_all_line_consistency(conv_dict):
    conv_dict['eval_all_line_consistency'] = []
    conv_dict['P1_all_line_consistency_score'] = 0
    conv_dict['P2_all_line_consistency_score'] = 0
    p1_utterances = 0
    p2_utterances = 0
    pturn = conv_dict["pturn"]
    for i, line in enumerate(conv_dict["conversation"]):
        if pturn == 1:
            prompt = prompts["eval_prompts"]["all_line_consistency"].replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                                                    .replace("%LISTENER_ROLE%", prompts["agent2_role"]) \
                                                                    .replace("%CONVERSATION%", "".join(conv_dict["conversation"])) \
                                                                    .replace("%SPEAKER_LINE%", line)
            output = completion_create(config['eval_model'], config, prompt)
            if config['verbose']:
                print(prompt)
            conv_dict['eval_all_line_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P1_all_line_consistency_score'] += 1
            p1_utterances += 1
            pturn = 2
        else:
            prompt = prompts["eval_prompts"]["all_line_consistency"].replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                                                                    .replace("%LISTENER_ROLE%", prompts["agent1_role"]) \
                                                                    .replace("%CONVERSATION%", "".join(conv_dict["conversation"])) \
                                                                    .replace("%SPEAKER_LINE%", line)
            output = completion_create(config['eval_model'], config, prompt)
            if config["verbose"]:
                print(prompt)
            conv_dict['eval_all_line_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P2_all_line_consistency_score'] += 1
            p2_utterances += 1
            pturn = 1

    if p1_utterances > 0:
        conv_dict['P1_all_line_consistency_score'] /= p1_utterances
    if p2_utterances > 0:
        conv_dict['P2_all_line_consistency_score'] /= p2_utterances


# (3) Survey of agent at every line (ANTHOLOGY ONLY FOR NOW)
def get_backstory_test(backstory, num_questions):
    ret = [[], []] # a list of questions, a list of answers
    for i in range(num_questions):
        prev_questions = ("" if len(ret[0]) == 0 else "\nFor reference, all of the following questions have already been asked:\n" + ''.join(ret[0]))
        prompt = prompts["eval_prompts"]["backstory_test"].replace("%SPEAKER_BACKSTORY%", backstory) \
                                                          .replace("%PREVIOUS_QUESTIONS%", prev_questions)
        if config["verbose"]:
            print(prompt)
        qa = completion_create(config['eval_model'], config, prompt)
        q = qa[qa.find('Question:'): qa.find('Answer:')]
        a = qa[qa.find('Answer:'): ]
        ret[0].append(q)
        ret[1].append(a)
    
    if config["verbose"]:
        print("BACKSTORY TEST")
        print(ret)
    return ret

def score_backstory_test(prompt, backstory_test):
    total_score = 0
    answers = []
    verdicts = []
    for i in range(len(backstory_test[0])):
        answer_prompt = prompt.replace("%BACKSTORY_QUESTION%", backstory_test[0][i])
        answer = completion_create(config['eval_model'], config, answer_prompt)

        verdict_prompt = prompts["eval_prompts"]["grade_backstory"].replace("%GIVEN_ANSWER%", answer) \
                                                                   .replace("%CORRECT_ANSWER%", backstory_test[1][i])
        if config["verbose"]:
            print(answer_prompt)
            print(verdict_prompt)
        verdict = completion_create(config['eval_model'], config, verdict_prompt)
        answers.append(answer)
        verdicts.append(verdict)
        score = 1 if 'yes' in verdict.lower() else 0
        # if score == 0:
            #print("WRONG!")
            #print("The prompt is\n" + prompt)
            # print("The correct answer to the question\n" + backstory_test[0][i] + "\nwas\n" + backstory_test[1][i] + "\nBut they answered\n" + answer)
        total_score += score
    return total_score, answers, verdicts

def eval_survey_consistency(conv_dict):
    p1_backstory = conv_dict["P1"]
    p2_backstory = conv_dict["P2"]
    p1_backstory_test = get_backstory_test(p1_backstory, 5)
    p2_backstory_test = get_backstory_test(p2_backstory, 5)
    
    conv_dict['eval_survey_consistency'] = []
    conv_dict['P1_survey_consistency_score'] = 0
    conv_dict['P2_survey_consistency_score'] = 0
    conv_dict['P1_backstory_test'] = p1_backstory_test
    conv_dict['P2_backstory_test'] = p2_backstory_test
    
    conversation = ""
    p1_utterances = 0
    p2_utterances = 0
    pturn = conv_dict["pturn"]
    for i, line in enumerate(conv_dict["conversation"]):
        conversation += line
        if pturn == 1:
            prompt = prompts["eval_prompts"]["answer_backstory"].replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                                                .replace("%LISTENER_ROLE%", prompts["agent2_role"]) \
                                                                .replace("%SPEAKER_BACKSTORY%", p1_backstory) \
                                                                .replace("%CONVERSATION%", conversation)
    
            score, answers, verdicts = score_backstory_test(prompt, p1_backstory_test)
            conv_dict['eval_survey_consistency'].append([line, score, answers, verdicts])
            conv_dict['P1_survey_consistency_score'] += score

            p1_utterances += 1
            pturn = 2
        else:
            prompt = prompts["eval_prompts"]["answer_backstory"].replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                                                                .replace("%LISTENER_ROLE%", prompts["agent1_role"]) \
                                                                .replace("%SPEAKER_BACKSTORY%", p2_backstory) \
                                                                .replace("%CONVERSATION%", conversation)

            score, answers, verdicts = score_backstory_test(prompt, p2_backstory_test)
            conv_dict['eval_survey_consistency'].append([line, score, answers, verdicts])
            conv_dict['P2_survey_consistency_score'] += score
            p2_utterances += 1
            pturn = 1
    if p1_utterances > 0:
        conv_dict['P1_survey_consistency_score'] /= p1_utterances
    if p2_utterances > 0:
        conv_dict['P2_survey_consistency_score'] /= p2_utterances
    

# (4) Takes in dialog, checks for inconsistency with previous line 
def eval_prev_line_consistency(conv_dict):
    conv_dict['eval_prev_line_consistency'] = []
    conv_dict['P1_prev_line_consistency_score'] = 0
    conv_dict['P2_prev_line_consistency_score'] = 0
    p1_utterances = 0
    p2_utterances = 0
    pturn = conv_dict["pturn"]
    for i, line in enumerate(conv_dict["conversation"]):
        if pturn == 1:
            prompt = prompts["eval_prompts"]["prev_line_consistency"].replace("%SPEAKER_ROLE%", prompts["agent1_role"]) \
                                                                     .replace("%LISTENER_ROLE%", prompts["agent2_role"]) \
                                                                     .replace("%CONVERSATION%", "".join(conv_dict["conversation"][:i])) \
                                                                     .replace("%SPEAKER_LINE%", line)
            if config['verbose']:
                print(prompt)
            output = completion_create(config['eval_model'], config, prompt)
            conv_dict['eval_prev_line_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P1_prev_line_consistency_score'] += 1
            p1_utterances += 1
            pturn = 2
        else:
            prompt = prompts["eval_prompts"]["prev_line_consistency"].replace("%SPEAKER_ROLE%", prompts["agent2_role"]) \
                                                                     .replace("%LISTENER_ROLE%", prompts["agent1_role"]) \
                                                                     .replace("%CONVERSATION%", "".join(conv_dict["conversation"][:i])) \
                                                                     .replace("%SPEAKER_LINE%", line)
            if config['verbose']:
                print(prompt)
            output = completion_create(config['eval_model'], config, prompt)
            conv_dict['eval_prev_line_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P2_prev_line_consistency_score'] += 1
            p2_utterances += 1
            pturn = 1

    if p1_utterances > 0:
        conv_dict['P1_prev_line_consistency_score'] /= p1_utterances
    if p2_utterances > 0:
        conv_dict['P2_prev_line_consistency_score'] /= p2_utterances

def run_metrics_short(filename):
    print(f"Begin metrics: {filename}\n\n")

    with open(filename, 'r') as f:
        conversations = json.load(f)

    for conversation in tqdm(conversations):
        if conversation['conversation_only']:
            if config['verbose']:
                print("BEGIN PROMPT CONSISTENCY")
            eval_prompt_consistency(conversation)
            if config['verbose']:
                print("BEGIN SURVEY CONSISTENCY")
            eval_survey_consistency(conversation)
            if config['verbose']:
                print("BEGIN PAIRWISE CONSISTENCY")
            eval_pairwise_consistency(conversation)
        conversation['conversation_only'] = False

    with open(filename, 'w') as f:
        json.dump(conversations, f, indent=4)
    
    print(f"End metrics: {filename}\n\n")

def run_metrics_all(filename):
    print(f"Begin metrics: {filename}\n\n")

    with open(filename, 'r') as f:
        conversations = json.load(f)

    for conversation in tqdm(conversations):
        if conversation['conversation_only']:
            if config['verbose']:
                print("BEGIN PROMPT CONSISTENCY")
            eval_prompt_consistency(conversation)
            if config['verbose']:
                print("BEGIN ALL LINE CONSISTENCY")
            eval_all_line_consistency(conversation)
            if config['verbose']:
                print("BEGIN PREVIOUS LINE CONSISTENCY")
            eval_prev_line_consistency(conversation)
            if config['verbose']:
                print("BEGIN SURVEY CONSISTENCY")
            eval_survey_consistency(conversation)
            if config['verbose']:
                print("BEGIN PAIRWISE CONSISTENCY")
            eval_pairwise_consistency(conversation)
        conversation['conversation_only'] = False

    with open(filename, 'w') as f:
        json.dump(conversations, f, indent=4)
    
    print(f"End metrics: {filename}\n\n")
    

def main(argv):
    global prompts
    init()
    config['eval_model'] = 'gpt-4o-mini' # we generally use gpt-4o-mini for evals 
    
    if config['task'] == 'Anthology':
        print("Using Anthology prompts")
        with open('config/persona_chat/prompts.json', 'r') as f:
            prompts = json.load(f)
        exp_folder = './data/anthology/exp'
    elif config['task'] == 'Education':
        print("Using Education prompts")
        with open('config/education/prompts.json', 'r') as f:
            prompts = json.load(f)
        exp_folder = './data/education/exp'
    
    # load general eval prompts
    with open('config/eval_prompts.json', 'r') as f:
        prompts['eval_prompts'] = json.load(f)

    if config['filename']:
        run_metrics_all(config['filename'])
    else:
        for filename in glob.glob(f'{exp_folder}/*.json'):
            run_metrics_all(filename)

if __name__ == '__main__':
    app.run(main)