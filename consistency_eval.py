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
            prompt = "For the following line spoken by P1, answer YES if the line contradicts the given backstory of P1, and answer NO if the line does not contradict the provided backstory of P1. P1's backstory is:\n" + conv_dict["P1"] + "\n P1 spoke the following line: \n" + line + "\n\n Answer YES if the line contradicts the given backstory of P1, and answer NO if the line does not contradict the provided backstory of P1, followed by 1 sentence of reasoning.\n\n"
            output = completion_create(config['eval_model'], config, prompt)
            conv_dict['eval_prompt_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P1_prompt_consistency_score'] += 1
            p1_utterances += 1
            pturn = 2
        else:
            prompt = "For the following line spoken by P2, answer YES if the line contradicts the given backstory of P2, and answer NO if the line does not contradict the provided backstory of P2. P2's backstory is:\n" + conv_dict["P2"] + "\n P2 spoke the following line: \n" + line + "\n\n Answer YES if the line contradicts the given backstory of P2, and answer NO if the line does not contradict the provided backstory of P2, followed by 1 sentence of reasoning.\n\n"
            output = completion_create(config['eval_model'], config, prompt)
            conv_dict['eval_prompt_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P2_prompt_consistency_score'] += 1
            p2_utterances += 1
            pturn = 1

    conv_dict['P1_prompt_consistency_score'] /= p1_utterances
    conv_dict['P2_prompt_consistency_score'] /= p2_utterances


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
            # old prompt (checks backstory as well)
            #prompt = "For the following line spoken by P1, answer YES if the line contradicts any line stated by P1 or P1's provided background, and answer NO if the line does not contradict any line in the provided conversation history of P1 and P1's provided background. P1 has the following backstory:\n" + conv_dict["P1"] + "\nP1 had the following conversation with P2:\n" + "".join(conv_dict["conversation"]) + "\n P1 spoke the following line: \n" + line + "\n\n Answer YES if the line contradicts any line stated by P1 throughout the course of the conversation or P1's provided background, and answer NO if the line does not contradict any line in the provided conversation history of P1 and P1's provided background, followed by 1 sentence of reasoning.\n\n"
            prompt = "For the following line spoken by P1, answer YES if the line contradicts any line stated by P1, and answer NO if the line does not contradict any line in the provided conversation history of P1. \nP1 had the following conversation with P2:\n" + "".join(conv_dict["conversation"]) + "\n P1 spoke the following line: \n" + line + "\n\n Answer YES if the line contradicts any line stated by P1 throughout the course of the conversation, and answer NO if the line does not contradict any line in the provided conversation history of P1, followed by 1 sentence of reasoning.\n\n"
            output = completion_create(config['eval_model'], config, prompt)
            conv_dict['eval_all_line_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P1_all_line_consistency_score'] += 1
            p1_utterances += 1
            pturn = 2
        else:
            #prompt = "For the following line spoken by P2, answer YES if the line contradicts any line stated by P2 or P2's provided background, and answer NO if the line does not contradict any line in the provided conversation history of P2 and P2's provided background. P2 has the following backstory:\n" + conv_dict["P2"] + "\nP2 had the following conversation with P1:\n" + "".join(conv_dict["conversation"]) + "\n P2 spoke the following line: \n" + line + "\n\n Answer YES if the line contradicts any line stated by P2 throughout the course of the conversation or P2's provided background, and answer NO if the line does not contradict any line in the provided conversation history of P2 and P2's provided background, followed by 1 sentence of reasoning.\n\n"
            prompt = "For the following line spoken by P2, answer YES if the line contradicts any line stated by P2, and answer NO if the line does not contradict any line in the provided conversation history of P2. \nP2 had the following conversation with P1:\n" + "".join(conv_dict["conversation"]) + "\n P2 spoke the following line: \n" + line + "\n\n Answer YES if the line contradicts any line stated by P2 throughout the course of the conversation, and answer NO if the line does not contradict any line in the provided conversation history of P2, followed by 1 sentence of reasoning.\n\n"
            output = completion_create(config['eval_model'], config, prompt)
            conv_dict['eval_all_line_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P2_all_line_consistency_score'] += 1
            p2_utterances += 1
            pturn = 1

    conv_dict['P1_all_line_consistency_score'] /= p1_utterances
    conv_dict['P2_all_line_consistency_score'] /= p2_utterances


# (3) Survey of agent at every line (ANTHOLOGY ONLY FOR NOW)
def get_backstory_test(backstory, num_questions):
    ret = [[], []] # a list of questions, a list of answers
    for i in range(num_questions):
        qa = completion_create(config['eval_model'], config, "Based on the following backstory, generate a new fact-based multiple choice question with 5 choices addressed directly IN SECOND PERSON, along with its correct answer. Preface the question with 'Question:' and the answer with 'Answer:'." + '\n' + backstory + ("" if len(ret[0]) == 0 else "\nFor reference, all of the following questions have already been asked:\n" + ''.join(ret[0])))
        q = qa[qa.find('Question:'): qa.find('Answer:')]
        a = qa[qa.find('Answer:'): ]
        ret[0].append(q)
        ret[1].append(a)
    return ret

def score_backstory_test(prompt, backstory_test):
    total_score = 0
    answers = []
    verdicts = []
    for i in range(len(backstory_test[0])):
        answer = completion_create(config['eval_model'], config, prompt + "\n Based on your conversation above so far, answer the following multiple choice question.\n" + backstory_test[0][i])
        verdict = completion_create(config['eval_model'], config, "As part of grading a test, determine whether the given answer " + answer + " matches the following correct answer. Respond with either YES or NO.\n" + "Correct " + backstory_test[1][i])
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
            prompt = "You are P1, and you are having a conversation with P2. Your backstory is:\n" + p1_backstory + "\n" + "So far, the conversation is as below:\n" + conversation

            score, answers, verdicts = score_backstory_test(prompt, p1_backstory_test, config)
            conv_dict['eval_survey_consistency'].append([line, score, answers, verdicts])
            conv_dict['P1_survey_consistency_score'] += score

            p1_utterances += 1
            pturn = 2
        else:
            prompt = "You are P2, and you are having a conversation with P1. Your backstory is:\n" + p2_backstory + "\n" + "So far, the conversation is as below:\n" + conversation

            score, answers, verdicts = score_backstory_test(prompt, p2_backstory_test, config)
            conv_dict['eval_survey_consistency'].append([line, score, answers, verdicts])
            conv_dict['P2_survey_consistency_score'] += score
            p2_utterances += 1
            pturn = 1
    conv_dict['P1_survey_consistency_score'] /= p1_utterances
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
            
            #prompt = "For the following line spoken by P1, answer YES if the line contradicts a previous line stated by P1 or P1's provided background, and answer NO if the line does not contradict the provided conversation history of P1 and P1's provided background. P1 has the following backstory:\n" + conv_dict["P1"] + "\nP1 had the following conversation with P2:\n" + "".join(conv_dict["conversation"][:i]) + "\n P1 spoke the following line: \n" + line + "\n\n Answer YES if the line contradicts a previous line stated by P1 or P1's provided background, and answer NO if the line does not contradict the provided conversation history of P1 and P1's provided background, followed by 1 sentence of reasoning.\n\n"
            prompt = "For the following line spoken by P1, answer YES if the line contradicts a previous line stated by P1, and answer NO if the line does not contradict the provided conversation history of P1. \nP1 had the following conversation with P2:\n" + "".join(conv_dict["conversation"][:i]) + "\n P1 spoke the following line: \n" + line + "\n\n Answer YES if the line contradicts a previous line stated by P1, and answer NO if the line does not contradict the provided conversation history of P1, followed by 1 sentence of reasoning.\n\n"
            output = completion_create(config['eval_model'], config, prompt)
            conv_dict['eval_prev_line_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P1_prev_line_consistency_score'] += 1
            p1_utterances += 1
            pturn = 2
        else:

            prompt = "For the following line spoken by P2, answer YES if the line contradicts a previous line stated by P2, and answer NO if the line does not contradict the provided conversation history of P2. \nP2 had the following conversation with P1:\n" + "".join(conv_dict["conversation"][:i]) + "\n P2 spoke the following line: \n" + line + "\n\n Answer YES if the line contradicts a previous line stated by P2, and answer NO if the line does not contradict the provided conversation history of P2, followed by 1 sentence of reasoning.\n\n"
            output = completion_create(config['eval_model'], config, prompt)
            conv_dict['eval_prev_line_consistency'].append(output)
            if "YES" not in output: # no contradiction
                conv_dict['P2_prev_line_consistency_score'] += 1
            p2_utterances += 1
            pturn = 1

    conv_dict['P1_prev_line_consistency_score'] /= p1_utterances
    conv_dict['P2_prev_line_consistency_score'] /= p2_utterances

def run_metrics(filename):
    print(f"\nBegin metrics: {filename}\n")

    with open(filename, 'r') as f:
        conversations = json.load(f)

    for conversation in conversations:
        eval_prompt_consistency(conversation)
        eval_all_line_consistency(conversation)
        eval_prev_line_consistency(conversation)
        eval_survey_consistency(conversation)

    with open(filename, 'w') as f:
        json.dump(conversations, f, indent=4)
    
    print(f"\nEnd metrics: {filename}\n")
    

def main(argv):
    init()
    config['eval_model'] = 'gpt-4o-mini' # we generally use gpt-4o-mini for evals 
    if config['filename']:
        run_metrics(config['filename'])
    else:
        for filename in glob.glob(f'{exp_folder}/*.json'):
            run_metrics(filename)

if __name__ == '__main__':
    app.run(main)